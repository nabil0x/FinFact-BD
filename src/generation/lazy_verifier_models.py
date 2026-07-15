from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Protocol

from src.generation.models import EmbeddingModel, SentenceTransformersEmbeddingModel
from src.generation.runtime import clear_cuda_cache

logger = logging.getLogger(__name__)


class LazyEmbeddingModel:
    def __init__(self, config: Dict[str, object], device: str) -> None:
        self._config = dict(config)
        self._device = device
        self._model: Optional[EmbeddingModel] = None
        self.model_name = str(self._config["model_name"])

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self._load().encode(texts)

    def release(self) -> None:
        self._model = None
        clear_cuda_cache()

    def _load(self) -> EmbeddingModel:
        if self._model is None:
            logger.info("Loading lazy embedding model %s", self.model_name)
            self._model = SentenceTransformersEmbeddingModel(
                self.model_name,
                device=self._device,
                prefix=str(self._config.get("prefix", "")),
            )
        return self._model


class LazyNLIModel:
    def __init__(self, config: Dict[str, object], device: str) -> None:
        self._config = dict(config)
        self._device = device
        self._model: Optional[BatchedTransformersNLIModel] = None
        self.model_name = str(self._config["model_name"])

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return self.contradiction_scores([premise], [hypothesis])[0]

    def contradiction_scores(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        return self._load().contradiction_scores(premises, hypotheses)

    def release(self) -> None:
        self._model = None
        clear_cuda_cache()

    def _load(self) -> "BatchedTransformersNLIModel":
        if self._model is None:
            logger.info("Loading lazy NLI model %s", self.model_name)
            self._model = BatchedTransformersNLIModel(
                self.model_name,
                revision=str(self._config.get("revision", "main")),
                device=self._device,
            )
        return self._model


class LazyFluencyModel:
    def __init__(self, config: Dict[str, object], device: str) -> None:
        self._config = dict(config)
        self._device = device
        self._model: Optional[BatchedFluencyModel] = None
        self.model_name = str(self._config["model_name"])

    def perplexity(self, text: str) -> float:
        return self.perplexities([text])[0]

    def perplexities(self, texts: List[str]) -> List[float]:
        return self._load().perplexities(texts)

    def release(self) -> None:
        self._model = None
        clear_cuda_cache()

    def _load(self) -> "BatchedFluencyModel":
        if self._model is None:
            backend = str(self._config.get("backend", "hf_causal_lm"))
            logger.info("Loading lazy fluency model %s backend=%s", self.model_name, backend)
            if backend == "hf_causal_lm":
                self._model = BatchedCausalLMFluencyModel(
                    self.model_name,
                    revision=str(self._config.get("revision", "main")),
                    device=self._device,
                )
            elif backend == "hf_electra_discriminator":
                self._model = BatchedElectraDiscriminatorQualityModel(
                    self.model_name,
                    revision=str(self._config.get("revision", "main")),
                    device=self._device,
                )
            else:
                raise ValueError(f"Unsupported fluency backend: {backend}")
        return self._model


class BatchedTransformersNLIModel:
    def __init__(self, model_name: str, revision: str = "main", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers for NLI verification") from exc
        self.model_name = model_name
        self._torch = torch
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).to(self._device)
        self._model.eval()
        self._contradiction_index = self._find_contradiction_index()
        logger.info("Loaded batched NLI model %s@%s", model_name, revision)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return self.contradiction_scores([premise], [hypothesis])[0]

    def contradiction_scores(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        if len(premises) != len(hypotheses):
            raise ValueError("NLI premise and hypothesis batches must have the same length")
        inputs = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)
        return [float(score) for score in probs[:, self._contradiction_index].detach().cpu().tolist()]

    def _find_contradiction_index(self) -> int:
        label2id = getattr(getattr(self._model, "config", None), "label2id", {}) or {}
        for label, idx in label2id.items():
            if "contrad" in str(label).lower():
                return int(idx)
        return 0


class BatchedFluencyModel(Protocol):
    model_name: str

    def perplexity(self, text: str) -> float:
        raise NotImplementedError

    def perplexities(self, texts: List[str]) -> List[float]:
        raise NotImplementedError


class BatchedCausalLMFluencyModel:
    def __init__(self, model_name: str, revision: str = "main", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers for fluency verification") from exc
        self.model_name = model_name
        self._torch = torch
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(self._device)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        logger.info("Loaded batched fluency model %s@%s", model_name, revision)

    def perplexity(self, text: str) -> float:
        return self.perplexities([text])[0]

    def perplexities(self, texts: List[str]) -> List[float]:
        inputs = self._tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self._device)
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = self._torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(shift_labels.size())
        mask = shift_labels.ne(-100)
        per_sample_loss = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return [float(math.exp(min(20.0, loss))) for loss in per_sample_loss.detach().cpu().tolist()]


class BatchedElectraDiscriminatorQualityModel:
    def __init__(self, model_name: str, revision: str = "main", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForPreTraining, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers for ELECTRA language-quality verification") from exc
        try:
            from normalizer import normalize
        except ImportError as exc:
            raise ImportError("Install csebuetnlp normalizer for BanglaBERT language-quality verification") from exc
        self.model_name = model_name
        self._torch = torch
        self._normalize = normalize
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForPreTraining.from_pretrained(model_name, revision=revision).to(self._device)
        self._model.eval()
        logger.info("Loaded batched ELECTRA quality model %s@%s", model_name, revision)

    def perplexity(self, text: str) -> float:
        return self.perplexities([text])[0]

    def perplexities(self, texts: List[str]) -> List[float]:
        normalized = [self._normalize(text) for text in texts]
        inputs = self._tokenizer(normalized, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
            fake_probs = self._torch.sigmoid(logits)
            mask = inputs["attention_mask"].bool()
            mean_fake = (fake_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return [float(1.0 + 300.0 * value) for value in mean_fake.detach().cpu().tolist()]
