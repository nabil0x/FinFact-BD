from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class GenerationModel(Protocol):
    model_name: str
    model_revision: str

    def generate_batch(
        self,
        prompts: List[str],
        temperatures: List[float],
        seeds: List[int],
        max_new_tokens: int,
    ) -> List[str]:
        """Generate one complete rewritten article per prompt."""


class EmbeddingModel(Protocol):
    model_name: str

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return dense sentence embeddings."""


class NLIModel(Protocol):
    model_name: str

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Return P(contradiction) for premise/hypothesis."""


class FluencyModel(Protocol):
    model_name: str

    def perplexity(self, text: str) -> float:
        """Return a language-model perplexity or equivalent fluency cost."""


@dataclass(frozen=True)
class ModelBundle:
    generator: GenerationModel
    embedder: EmbeddingModel
    nli: NLIModel
    fluency: FluencyModel


class HuggingFaceSeq2SeqGenerator:
    def __init__(self, model_name: str, revision: str = "main", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to use hf_seq2seq generation") from exc
        self.model_name = model_name
        self.model_revision = revision
        self._torch = torch
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision).to(self._device)
        self._model.eval()
        logger.info("Loaded seq2seq generator %s@%s on %s", model_name, revision, self._device)

    def generate_batch(
        self,
        prompts: List[str],
        temperatures: List[float],
        seeds: List[int],
        max_new_tokens: int,
    ) -> List[str]:
        outputs: List[str] = []
        for prompt, temperature, seed in zip(prompts, temperatures, seeds):
            self._torch.manual_seed(seed)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self._device)
            with self._torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    num_beams=1 if temperature > 0.0 else 4,
                    early_stopping=True,
                )
            outputs.append(self._tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        return outputs


class SentenceTransformersEmbeddingModel:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers for embedding verification") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("Loaded embedding model %s", model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]


class TransformersNLIModel:
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
        logger.info("Loaded NLI model %s@%s", model_name, revision)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
        return float(probs[self._contradiction_index].item())

    def _find_contradiction_index(self) -> int:
        label2id = getattr(getattr(self._model, "config", None), "label2id", {}) or {}
        for label, idx in label2id.items():
            if "contrad" in str(label).lower():
                return int(idx)
        return 0


class CausalLMFluencyModel:
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
        self._model.eval()
        logger.info("Loaded fluency model %s@%s", model_name, revision)

    def perplexity(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self._device)
        with self._torch.no_grad():
            loss = self._model(**inputs, labels=inputs["input_ids"]).loss
        return float(math.exp(min(20.0, loss.item())))


def build_model_bundle(config: Dict[str, object]) -> ModelBundle:
    device = str(config.get("device", "cuda"))
    generator_cfg = dict(config["generator"])  # type: ignore[index]
    embedding_cfg = dict(config["embedding"])  # type: ignore[index]
    nli_cfg = dict(config["nli"])  # type: ignore[index]
    fluency_cfg = dict(config["fluency"])  # type: ignore[index]
    generator = _build_generator(generator_cfg, device)
    embedder = SentenceTransformersEmbeddingModel(str(embedding_cfg["model_name"]), device=device)
    nli = TransformersNLIModel(
        str(nli_cfg["model_name"]),
        revision=str(nli_cfg.get("revision", "main")),
        device=device,
    )
    fluency = CausalLMFluencyModel(
        str(fluency_cfg["model_name"]),
        revision=str(fluency_cfg.get("revision", "main")),
        device=device,
    )
    return ModelBundle(generator=generator, embedder=embedder, nli=nli, fluency=fluency)


def _build_generator(config: Dict[str, object], device: str) -> GenerationModel:
    backend = str(config.get("backend", "hf_seq2seq"))
    if backend == "hf_seq2seq":
        return HuggingFaceSeq2SeqGenerator(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
        )
    raise ValueError(f"Unsupported generator backend: {backend}")
