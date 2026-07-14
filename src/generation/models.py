from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

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


class InstructionModel(Protocol):
    model_name: str
    model_revision: str

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        """Generate one instruction-following text response."""


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
    extractor: Optional[InstructionModel] = None
    planner: Optional[InstructionModel] = None


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

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        return self.generate_batch([prompt], [temperature], [seed], max_new_tokens)[0]


class HuggingFaceCausalLMGenerator:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cuda",
        load_in_4bit: bool = False,
        use_chat_template: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to use hf_causal_lm generation") from exc
        self.model_name = model_name
        self.model_revision = revision
        self._torch = torch
        self._use_chat_template = use_chat_template
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        kwargs: Dict[str, Any] = {"revision": revision, "trust_remote_code": True}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("Install bitsandbytes-compatible transformers for 4-bit loading") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit:
            self._model = self._model.to(self._device)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        logger.info("Loaded causal LM %s@%s 4bit=%s", model_name, revision, load_in_4bit)

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
            formatted = self._format_prompt(prompt)
            inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with self._torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            outputs.append(self._tokenizer.decode(ids[0][prompt_len:], skip_special_tokens=True).strip())
        return outputs

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        return self.generate_batch([prompt], [temperature], [seed], max_new_tokens)[0]

    def _format_prompt(self, prompt: str) -> str:
        if self._use_chat_template and getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt


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


class MaskedLMFluencyModel:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cuda",
        max_masked_tokens: int = 96,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers for masked-LM fluency verification") from exc
        self.model_name = model_name
        self._torch = torch
        self._max_masked_tokens = max_masked_tokens
        self._device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision).to(self._device)
        if self._tokenizer.mask_token_id is None:
            raise ValueError(f"Masked-LM fluency model lacks a mask token: {model_name}")
        self._model.eval()
        logger.info("Loaded masked-LM fluency model %s@%s", model_name, revision)

    def perplexity(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self._device)
        ids = inputs["input_ids"][0]
        special_ids = set(self._tokenizer.all_special_ids)
        candidate_positions = [idx for idx, token_id in enumerate(ids.tolist()) if token_id not in special_ids]
        candidate_positions = candidate_positions[: self._max_masked_tokens]
        if not candidate_positions:
            return 999.0
        losses = []
        with self._torch.no_grad():
            for idx in candidate_positions:
                masked = ids.clone()
                labels = self._torch.full_like(masked, -100)
                labels[idx] = ids[idx]
                masked[idx] = self._tokenizer.mask_token_id
                loss = self._model(
                    input_ids=masked.unsqueeze(0),
                    attention_mask=inputs["attention_mask"],
                    labels=labels.unsqueeze(0),
                ).loss
                losses.append(float(loss.item()))
        return float(math.exp(min(20.0, sum(losses) / len(losses))))


def build_model_bundle(config: Dict[str, object]) -> ModelBundle:
    device = str(config.get("device", "cuda"))
    generator_cfg = dict(config["generator"])  # type: ignore[index]
    embedding_cfg = dict(config["embedding"])  # type: ignore[index]
    nli_cfg = dict(config["nli"])  # type: ignore[index]
    fluency_cfg = dict(config["fluency"])  # type: ignore[index]
    generator = _build_generator(generator_cfg, device)
    extractor = _build_optional_instruction_model(dict(config.get("extractor", {})), device, {})
    shared = {"extractor": extractor}
    planner = _build_optional_instruction_model(dict(config.get("planner", {})), device, shared)
    embedder = SentenceTransformersEmbeddingModel(str(embedding_cfg["model_name"]), device=device)
    nli = TransformersNLIModel(
        str(nli_cfg["model_name"]),
        revision=str(nli_cfg.get("revision", "main")),
        device=device,
    )
    fluency = _build_fluency_model(fluency_cfg, device)
    return ModelBundle(
        generator=generator,
        embedder=embedder,
        nli=nli,
        fluency=fluency,
        extractor=extractor,
        planner=planner,
    )


def _build_generator(config: Dict[str, object], device: str) -> GenerationModel:
    backend = str(config.get("backend", "hf_seq2seq"))
    if backend == "hf_seq2seq":
        return HuggingFaceSeq2SeqGenerator(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
        )
    if backend == "hf_causal_lm":
        return HuggingFaceCausalLMGenerator(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
            load_in_4bit=bool(config.get("load_in_4bit", False)),
            use_chat_template=bool(config.get("use_chat_template", True)),
        )
    raise ValueError(f"Unsupported generator backend: {backend}")


def _build_fluency_model(config: Dict[str, object], device: str) -> FluencyModel:
    backend = str(config.get("backend", "hf_causal_lm"))
    if backend == "hf_causal_lm":
        return CausalLMFluencyModel(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
        )
    if backend == "hf_masked_lm":
        return MaskedLMFluencyModel(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
            max_masked_tokens=int(config.get("max_masked_tokens", 96)),
        )
    raise ValueError(f"Unsupported fluency backend: {backend}")


def _build_optional_instruction_model(
    config: Dict[str, object],
    device: str,
    shared: Dict[str, Optional[InstructionModel]],
) -> Optional[InstructionModel]:
    if not config or not bool(config.get("enabled", False)):
        return None
    share_with = config.get("share_with")
    if share_with:
        model = shared.get(str(share_with))
        if model is None:
            raise ValueError(f"Cannot share instruction model with unavailable role: {share_with}")
        return model
    return _build_generator(config, device)  # type: ignore[return-value]
