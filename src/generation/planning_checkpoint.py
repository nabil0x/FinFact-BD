from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.generation.metadata import Article, Claim, PlannedArticle, RankedClaim, RewritePlan

logger = logging.getLogger(__name__)


def load_planned_articles(path: Path) -> Dict[str, PlannedArticle]:
    if not path.exists():
        return {}
    planned: Dict[str, PlannedArticle] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                item = planned_article_from_dict(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed planned article row path=%s line=%d error=%s", path, line_number, exc)
                continue
            planned[item.article.article_id] = item
    logger.info("Loaded %d planned articles from %s", len(planned), path)
    return planned


def append_planned_article(path: Path, planned: PlannedArticle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(planned_article_to_dict(planned), ensure_ascii=False) + "\n")


def planned_article_to_dict(planned: PlannedArticle) -> Dict[str, Any]:
    return {
        "article": {
            "article_id": planned.article.article_id,
            "headline": planned.article.headline,
            "text": planned.article.text,
            "metadata": planned.article.metadata,
        },
        "selected": planned.selected.to_dict(),
        "plan": planned.plan.to_dict(),
        "sample_seed": planned.sample_seed,
    }


def planned_article_from_dict(row: Dict[str, Any]) -> PlannedArticle:
    article = Article(**row["article"])
    selected = _ranked_claim_from_dict(row["selected"])
    plan = _rewrite_plan_from_dict(row["plan"])
    return PlannedArticle(article=article, selected=selected, plan=plan, sample_seed=int(row["sample_seed"]))


def _ranked_claim_from_dict(row: Dict[str, Any]) -> RankedClaim:
    data = dict(row)
    data["claim"] = Claim(**data["claim"])
    return RankedClaim(**data)


def _rewrite_plan_from_dict(row: Dict[str, Any]) -> RewritePlan:
    data = dict(row)
    data["target_claim"] = Claim(**data["target_claim"])
    return RewritePlan(**data)
