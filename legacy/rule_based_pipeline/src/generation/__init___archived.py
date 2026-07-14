"""FinFact-BD generation package — claim-guided rewriting pipeline."""

from src.generation.claim_extraction import (
    ExtractedClaim,
    extract_claims,
    split_sentences,
)
from src.generation.claim_selection import (
    SelectedClaim,
    select_claim,
)
from src.generation.claim_planning import (
    CLAIM_TYPE_TO_FAMILY,
    DESIRED_CHANGE_TEMPLATES,
    EXPECTED_CHANGED_TEMPLATES,
    FAMILY_TO_CLAIM_TYPE,
    SCOPE_BY_FAMILY,
    RewritePlan,
    create_rewrite_plan,
)
from src.generation.bangla_rewriter import BanglaRewriter
from src.generation.verification import (
    StageResult,
    VerificationResult,
    RewriteVerifier,
)
from src.generation.pipeline import (
    ClaimGuidedRewritePipeline,
    PipelineResult,
)
from src.generation.export import export_results

__all__ = [
    # Claim extraction
    "ExtractedClaim",
    "extract_claims",
    "split_sentences",
    # Claim selection
    "SelectedClaim",
    "select_claim",
    # Claim planning
    "CLAIM_TYPE_TO_FAMILY",
    "DESIRED_CHANGE_TEMPLATES",
    "EXPECTED_CHANGED_TEMPLATES",
    "FAMILY_TO_CLAIM_TYPE",
    "SCOPE_BY_FAMILY",
    "RewritePlan",
    "create_rewrite_plan",
    # Rewriting
    "BanglaRewriter",
    # Verification
    "StageResult",
    "VerificationResult",
    "RewriteVerifier",
    # Pipeline
    "ClaimGuidedRewritePipeline",
    "PipelineResult",
    "export_results",
]
