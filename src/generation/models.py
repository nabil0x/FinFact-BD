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


class HuggingFaceCausalLMGenerator:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cuda",
        load_in_4bit: bool = False,
        use_chat_template: bool = True,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
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
        self._chat_template_kwargs = chat_template_kwargs or {}
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
                generate_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0.0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                }
                if temperature > 0.0:
                    generate_kwargs["temperature"] = max(temperature, 1e-5)
                ids = self._model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            outputs.append(self._tokenizer.decode(ids[0][prompt_len:], skip_special_tokens=True).strip())
        return outputs

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        return self.generate_batch([prompt], [temperature], [seed], max_new_tokens)[0]

    def _format_prompt(self, prompt: str) -> str:
        if self._use_chat_template and getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )
        return prompt


class LazyGenerationModel:
    def __init__(self, config: Dict[str, object], device: str) -> None:
        self._config = dict(config)
        self._device = device
        self._model: Optional[GenerationModel] = None
        self._unload_after_call = bool(self._config.get("unload_after_call", False))
        self.model_name = str(self._config["model_name"])
        self.model_revision = str(self._config.get("revision", "main"))

    def generate_batch(
        self,
        prompts: List[str],
        temperatures: List[float],
        seeds: List[int],
        max_new_tokens: int,
    ) -> List[str]:
        model = self._load()
        try:
            return model.generate_batch(prompts, temperatures, seeds, max_new_tokens)
        finally:
            if self._unload_after_call:
                self._unload()

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        return self.generate_batch([prompt], [temperature], [seed], max_new_tokens)[0]

    def _load(self) -> GenerationModel:
        if self._model is None:
            logger.info("Loading lazy generation model %s@%s", self.model_name, self.model_revision)
            cfg = dict(self._config)
            cfg["lazy"] = False
            self._model = _build_generator(cfg, self._device)
        return self._model

    def release(self) -> None:
        self._unload()

    def _unload(self) -> None:
        self._model = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return


class SentenceTransformersEmbeddingModel:
    def __init__(self, model_name: str, device: str = "cuda", prefix: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers for embedding verification") from exc
        self.model_name = model_name
        self._prefix = prefix
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("Loaded embedding model %s", model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = [f"{self._prefix}{text}" if self._prefix else text for text in texts]
        vectors = self._model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
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


class ElectraDiscriminatorQualityModel:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cuda",
    ) -> None:
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
        logger.info("Loaded ELECTRA quality model %s@%s", model_name, revision)

    def perplexity(self, text: str) -> float:
        normalized = self._normalize(text)
        inputs = self._tokenizer(normalized, return_tensors="pt", truncation=True, max_length=512).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            fake_probs = self._torch.sigmoid(logits)
            mask = inputs["attention_mask"].bool()
            mean_fake = fake_probs[mask].mean().item()
        return float(1.0 + 300.0 * mean_fake)


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
    embedder = SentenceTransformersEmbeddingModel(
        str(embedding_cfg["model_name"]),
        device=device,
        prefix=str(embedding_cfg.get("prefix", "")),
    )
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
    if bool(config.get("lazy", False)):
        return LazyGenerationModel(config, device)
    backend = str(config.get("backend", "hf_causal_lm"))
    if backend == "hf_causal_lm":
        return HuggingFaceCausalLMGenerator(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
            load_in_4bit=bool(config.get("load_in_4bit", False)),
            use_chat_template=bool(config.get("use_chat_template", True)),
            chat_template_kwargs=dict(config.get("chat_template_kwargs", {})),
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
    if backend == "hf_electra_discriminator":
        return ElectraDiscriminatorQualityModel(
            str(config["model_name"]),
            revision=str(config.get("revision", "main")),
            device=device,
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
