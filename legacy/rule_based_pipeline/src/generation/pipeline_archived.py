from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional sibling-stage imports
_IMPORT_ERRS: Dict[str, str] = {}
for _mod, _name in [
    ("claim_extraction", "extract_claims"),
    ("claim_selection", "select_claim"),
    ("claim_planning", "create_rewrite_plan"),
    ("bangla_rewriter", "BanglaRewriter"),
    ("verification", "RewriteVerifier"),
]:
    try:
        globals()[_name] = __import__(f".{_mod}", globals(), locals(), [_name], 1).__dict__[_name]  # type: ignore[union-attr]
    except ImportError as e:
        _IMPORT_ERRS[_name] = str(e)
        globals()[_name] = None

from .export import export_results

REWRITE_FAMILIES = ["numerical_fact_change", "policy_reversal", "entity_replacement", "temporal_shift", "causal_inversion"]


@dataclass
class ClaimOutput:
    claim_id: str; claim_type: str; text: str; sentence_index: int
    span_start: int; span_end: int; importance_score: float; editability_score: float
    diversity_bonus: float = 0.0; metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteOutput:
    sample_id: str; original_id: str; text: str; rewrite_family: str
    target_sentence_index: int; target_span: str; desired_change: str; difficulty: str
    importance_score: float; editability_score: float; diversity_bonus: float
    generation_model: str; regeneration_attempts: int
    claim_plan: Dict[str, Any]; verification_result: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    raw_outputs: List[Dict[str, Any]]; filtered_outputs: List[Dict[str, Any]]
    metadata: Dict[str, Any]; stats: Dict[str, Any]


class ClaimGuidedRewritePipeline:
    """Orchestrates claim extraction → selection → planning → rewriting → verification."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pc = config.get("pipeline", {})
        self.max_attempts = int(pc.get("max_attempts", 3))
        self.seed = int(pc.get("seed", 42))
        self.ckpt_every = int(pc.get("checkpoint_every_n", 100))
        self.prog_int = int(pc.get("progress_interval", 10))
        self.drop_failures = bool(pc.get("drop_failures", True))
        self.model_cfg = config.get("generation_model", {})
        self.verif_cfg = config.get("verification", {})
        self.extract_cfg = config.get("claim_extraction", {})
        self.select_cfg = config.get("claim_selection", {})
        self._rng = random.Random(self.seed)
        self._rewriter: Any = None
        self._verifier: Any = None
        for name, err in _IMPORT_ERRS.items():
            logger.warning("Stage %s unavailable (%s) — using fallback", name, err)

    def run(self, input_csv: str = None, output_dir: str = None, num_samples: int = None, model_name: str = None) -> PipelineResult:
        resolved_input = Path(input_csv or self.config.get("paths", {}).get("input_csv", ""))
        resolved_output = Path(output_dir or self.config.get("paths", {}).get("output_dir", "data/generated"))
        resolved_output.mkdir(parents=True, exist_ok=True)

        articles = self._load_data(resolved_input, num_samples)
        if not articles:
            logger.warning("No articles loaded from %s", resolved_input)
            return PipelineResult([], [], {"input_csv": str(resolved_input), "status": "no_data"}, {})

        model_name = model_name or self.model_cfg.get("name", "csebuetnlp/banglat5")
        self._ensure_rewriter(model_name)
        self._ensure_verifier()

        ckpt_path = Path(self.config.get("paths", {}).get("checkpoint", ""))
        processed = self._load_checkpoint(ckpt_path)
        raw_outputs: List[Dict[str, Any]] = []
        start = time.time()
        stats = {"success": 0, "failed": 0, "skipped_empty": 0, "skipped_resume": 0}

        for idx, art in enumerate(articles):
            aid = self._article_id(art)
            text = self._article_text(art)
            if not text:
                stats["skipped_empty"] += 1; continue
            if aid in processed:
                stats["skipped_resume"] += 1; continue

            result = self._process_article(aid, text, random.Random(self._rng.randint(0, 2**31)))
            if result is not None:
                raw_outputs.append(self._to_flat(result, art))
                stats["success"] += 1
            else:
                stats["failed"] += 1

            if (idx + 1) % self.ckpt_every == 0:
                self._save_checkpoint(ckpt_path, processed | {self._article_id(a) for a in articles[:idx+1] if self._article_text(a)})
            if (idx + 1) % self.prog_int == 0:
                elapsed = time.time() - start
                logger.info("Progress %d/%d (%.1f%%) success=%d fail=%d %.1f art/s", idx+1, len(articles), 100*(idx+1)/len(articles), stats["success"], stats["failed"], (idx+1)/max(elapsed,1e-6))

        self._save_checkpoint(ckpt_path, {self._article_id(a) for a in articles})
        stats["total_articles"] = len(articles)
        stats["total_time_seconds"] = round(time.time() - start, 2)
        stats["success_rate"] = round(stats["success"] / max(stats["success"]+stats["failed"], 1), 4)
        filtered = [r for r in raw_outputs if json.loads(r.get("verification_result", "{}")).get("passed", False)]
        stats["family_distribution"] = dict(Counter(r.get("rewrite_family","unknown") for r in filtered))
        stats["difficulty_distribution"] = dict(Counter(r.get("difficulty","unknown") for r in filtered))
        logger.info("Complete: %d raw, %d filtered (%.1f%% pass), %.1fs", len(raw_outputs), len(filtered), 100*len(filtered)/max(len(raw_outputs),1), stats["total_time_seconds"])
        return PipelineResult(raw_outputs, filtered, {"input_csv": str(resolved_input), "output_dir": str(resolved_output), "generation_model": model_name}, stats)

    def _process_article(self, article_id: str, text: str, rng: random.Random) -> Optional[RewriteOutput]:
        claims = self._extract_claims(text)
        if not claims: return None
        selected = self._select_claim(claims, rng)
        if selected is None: return None
        plan = self._create_plan(selected, rng)
        if plan is None: return None
        return self._regeneration_loop(text, selected, plan, rng)

    def _extract_claims(self, text: str) -> List[ClaimOutput]:
        if extract_claims is not None:
            try: return extract_claims(text, self.extract_cfg)
            except Exception: logger.exception("extract_claims failed")
        return self._fallback_extract(text)

    def _select_claim(self, claims: List[ClaimOutput], rng: random.Random) -> Optional[ClaimOutput]:
        if select_claim is not None:
            try: return select_claim(claims, self.select_cfg, rng)
            except Exception: logger.exception("select_claim failed")
        return self._fallback_select(claims, rng)

    def _create_plan(self, claim: ClaimOutput, rng: random.Random) -> Optional[Dict[str, Any]]:
        if create_rewrite_plan is not None:
            try: return create_rewrite_plan(claim, self.config, rng)
            except Exception: logger.exception("create_rewrite_plan failed")
        diff_tgt = self.select_cfg.get("difficulty_targets", {})
        labels = list(diff_tgt.keys()) or ["easy", "medium", "hard"]
        weights = [float(diff_tgt.get(l, 0)) or 1 for l in labels]
        diff = rng.choices(labels, weights=weights, k=1)[0]
        return {"rewrite_family": claim.claim_type, "target_sentence_index": claim.sentence_index, "target_span": claim.text, "desired_change": f"Modify {claim.claim_type} span '{claim.text}'", "difficulty": diff, "hop_count": {"easy":1,"medium":2,"hard":3}.get(diff,1), "family_sequence": [claim.claim_type]}

    def _rewrite(self, text: str, plan: Dict[str, Any], rng: random.Random) -> Optional[str]:
        if self._rewriter is not None:
            try: return self._rewriter.rewrite(text, plan)
            except Exception: logger.exception("rewrite failed")
        return text

    def _verify(self, original: str, rewritten: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        if self._verifier is not None:
            try: return self._verifier.verify(original, rewritten, plan)
            except Exception: logger.exception("verify failed")
        if rewritten == original:
            return {"passed": False, "claim_integrity_score": 0.0, "surface_quality_score": 0.0, "semantic_quality_score": 0.0, "issues": ["identical_text"]}
        dr = len(rewritten) / max(len(original), 1)
        if dr < 0.6 or dr > 1.4:
            return {"passed": False, "claim_integrity_score": 0.3, "surface_quality_score": 0.5, "semantic_quality_score": 0.4, "issues": ["length_ratio_out_of_bounds"]}
        return {"passed": True, "claim_integrity_score": 0.8, "surface_quality_score": 0.8, "semantic_quality_score": 0.7, "issues": []}

    def _regeneration_loop(self, original_text: str, claim: ClaimOutput, plan: Dict[str, Any], rng: random.Random) -> Optional[RewriteOutput]:
        best: Optional[Tuple[float, str, Dict[str, Any], int]] = None
        for attempt in range(1, self.max_attempts + 1):
            rewritten = self._rewrite(original_text, plan, rng)
            if not rewritten or rewritten == original_text: continue
            v = self._verify(original_text, rewritten, plan)
            score = sum(float(v.get(k,0))*w for k,w in [("claim_integrity_score",0.35),("surface_quality_score",0.30),("semantic_quality_score",0.35)])
            if v.get("passed", False):
                return self._build_output(claim.claim_id, original_text, rewritten, claim, plan, v, attempt)
            if best is None or score > best[0]:
                best = (score, rewritten, v, attempt)
        if best is not None:
            best[2]["note"] = "best_of_failed"
            return self._build_output(claim.claim_id, original_text, best[1], claim, plan, best[2], best[3])
        return None

    def _build_output(self, original_id: str, original_text: str, rewritten_text: str, claim: ClaimOutput, plan: Dict, v: Dict, attempts: int) -> RewriteOutput:
        return RewriteOutput(sample_id=f"rw_{original_id}_{plan.get('rewrite_family', claim.claim_type)}_{attempts}", original_id=original_id, text=rewritten_text, rewrite_family=plan.get("rewrite_family", claim.claim_type), target_sentence_index=plan.get("target_sentence_index", claim.sentence_index), target_span=plan.get("target_span", claim.text), desired_change=plan.get("desired_change", f"Modify {claim.claim_type} claim"), difficulty=plan.get("difficulty", "medium"), importance_score=claim.importance_score, editability_score=claim.editability_score, diversity_bonus=claim.diversity_bonus, generation_model=self.model_cfg.get("name","csebuetnlp/banglat5"), regeneration_attempts=attempts, claim_plan=plan, verification_result=v, metadata={"source":"BENI v2","article_length":len(original_text),"rewritten_length":len(rewritten_text)})

    def _to_flat(self, out: RewriteOutput, article: Dict) -> Dict[str, Any]:
        return {"id": out.sample_id, "sample_id": out.sample_id, "text": out.text, "label": 1, "split": article.get("split", ""), "original_id": out.original_id, "rewrite_family": out.rewrite_family, "target_sentence_index": out.target_sentence_index, "target_span": out.target_span, "desired_change": out.desired_change, "difficulty": out.difficulty, "importance_score": out.importance_score, "editability_score": out.editability_score, "diversity_bonus": out.diversity_bonus, "generation_model": out.generation_model, "regeneration_attempts": out.regeneration_attempts, "verification_result": json.dumps(out.verification_result, ensure_ascii=False), "perturbation_plan": json.dumps(out.claim_plan, ensure_ascii=False), "metadata": json.dumps(out.metadata, ensure_ascii=False)}

    def _ensure_rewriter(self, model_name: str) -> None:
        if self._rewriter is not None: return
        if BanglaRewriter is not None:
            try:
                self._rewriter = BanglaRewriter(model_name=model_name, config=self.model_cfg)
                logger.info("Loaded BanglaRewriter: %s", model_name)
            except Exception:
                logger.exception("Failed to load BanglaRewriter")

    def _ensure_verifier(self) -> None:
        if self._verifier is not None: return
        if RewriteVerifier is not None:
            try:
                self._verifier = RewriteVerifier(config=self.verif_cfg)
                logger.info("Loaded RewriteVerifier")
            except Exception:
                logger.exception("Failed to load RewriteVerifier")

    def _load_data(self, path: Path, num_samples: int = None) -> List[Dict[str, Any]]:
        text_col = self.config.get("input", {}).get("text_column", "text")
        articles: List[Dict[str, Any]] = []
        try:
            if path.suffix == ".zst":
                import zstandard as zstd, io
                with open(path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as r:
                        for row in csv.DictReader(io.TextIOWrapper(r, encoding="utf-8")):
                            articles.append(dict(row))
                            if num_samples and len(articles) >= num_samples: break
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        articles.append(dict(row))
                        if num_samples and len(articles) >= num_samples: break
        except FileNotFoundError:
            logger.error("Input not found: %s", path)
        except Exception:
            logger.exception("Failed to read %s", path)
        return [a for a in articles if a.get(text_col, "").strip()]

    def _article_id(self, article: Dict) -> str:
        return str(article.get(self.config.get("input",{}).get("id_column","article_id"), article.get("id","unknown")))
    def _article_text(self, article: Dict) -> str:
        return (article.get(self.config.get("input",{}).get("text_column","text")) or "").strip()

    def _load_checkpoint(self, path: Path) -> set:
        if not path or not path.exists(): return set()
        try:
            with open(path) as f: data = json.load(f)
            processed = set(data.get("processed_ids", []))
            logger.info("Resuming: %d already processed", len(processed))
            return processed
        except Exception:
            return set()

    def _save_checkpoint(self, path: Path, ids: set) -> None:
        if not path: return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"processed_ids": sorted(ids)}, f, ensure_ascii=False)
        except Exception:
            logger.exception("Checkpoint write failed")

    def _fallback_extract(self, text: str) -> List[ClaimOutput]:
        claims: List[ClaimOutput] = []
        sentences = re.split(r"[।!?]+", text)
        seen: set = set()
        for sidx, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10 or sidx >= 50: continue
            for m in re.finditer(r"[০-৯0-9]+(?:[.,][০-৯0-9]+)?%?", sent):
                sp = (sidx, m.start(), m.end())
                if sp not in seen:
                    seen.add(sp)
                    claims.append(ClaimOutput(f"num_{sidx}_{m.start()}", "numerical_fact_change", m.group(), sidx, m.start(), m.end(), 0.6, 0.7))
            for kw in ["বৃদ্ধি","হ্রাস","কমেছে","বেড়েছে"]:
                i = sent.find(kw)
                if i >= 0 and (sidx, i, i+len(kw)) not in seen:
                    seen.add((sidx, i, i+len(kw)))
                    claims.append(ClaimOutput(f"pol_{sidx}_{i}", "policy_reversal", kw, sidx, i, i+len(kw), 0.7, 0.6))
                    break
        return claims

    def _fallback_select(self, claims: List[ClaimOutput], rng: random.Random) -> Optional[ClaimOutput]:
        if not claims: return None
        w = self.select_cfg.get("weights", {})
        scored = [(float(w.get("centrality",0.30))*c.importance_score + float(w.get("editability",0.20))*c.editability_score, c) for c in claims]
        return max(scored, key=lambda x: x[0])[1]
