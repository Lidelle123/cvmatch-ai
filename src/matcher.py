"""
Matching engine — compares a CV to a job offer using LLM-as-judge.

Why LLM-as-judge instead of pure embeddings cosine similarity?
- Cosine similarity gives ONE number, no explanation.
- LLM-as-judge gives a score PLUS structured rationale (matched skills,
  gaps, strengths, weaknesses), which is what recruiters and candidates need.
- The token cost is acceptable for one-shot analysis per candidacy.
"""
import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from src.llm_client import chat
from src.cv_parser import CVData, extract_json_from_text
from src.job_parser import JobOffer


# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────

class MatchingResult(BaseModel):
    overall_score: int = Field(
        ge=0, le=100,
        description="Global compatibility score (0-100)"
    )
    score_breakdown: dict = Field(
        default_factory=dict,
        description="Sub-scores by dimension: skills, experience, education (each 0-100)"
    )

    skills_matched: list[str] = Field(default_factory=list)
    skills_missing: list[str] = Field(default_factory=list)
    skills_bonus: list[str] = Field(
        default_factory=list,
        description="CV skills matching nice-to-have offer skills"
    )

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    experience_assessment: str = Field(
        description="Qualitative analysis of how the candidate's experience aligns"
    )
    recommendations: list[str] = Field(default_factory=list)

    verdict: Literal["strong_match", "good_match", "partial_match", "weak_match"]


# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un recruteur expert et un coach carrière exigeant mais juste.
Ton rôle est d'évaluer rigoureusement la compatibilité entre un CV et une offre.

PRINCIPES D'ÉVALUATION :
1. Sois HONNÊTE et FACTUEL : ni trop optimiste, ni trop dur.
2. Base-toi UNIQUEMENT sur les informations fournies. Pas de suppositions.
3. Un score > 80 implique : majorité des compétences requises + expérience pertinente.
4. Distingue compétences techniques, soft skills, expérience, et formation.
5. Réponds UNIQUEMENT en JSON valide.

VERDICTS :
- "strong_match"  : 85-100  → candidat très bien aligné
- "good_match"    : 70-84   → bon profil, gaps mineurs
- "partial_match" : 50-69   → match partiel, gaps significatifs
- "weak_match"    : 0-49    → décalage important

Réponds dans la langue de l'offre d'emploi.
"""


def _build_user_prompt(cv: CVData, job: JobOffer) -> str:
    schema = MatchingResult.model_json_schema()
    return f"""Compare ce CV à cette offre d'emploi.

=== CV DU CANDIDAT ===
{cv.model_dump_json(indent=2)}

=== OFFRE D'EMPLOI ===
{job.model_dump_json(indent=2)}

Analyse rigoureusement la compatibilité avec ce schéma EXACT :

{json.dumps(schema, indent=2, ensure_ascii=False)}

Réponds UNIQUEMENT avec le JSON, sans markdown, sans commentaire.
"""


# ─────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────

def match_cv_to_job(cv: CVData, job: JobOffer, max_retries: int = 2) -> MatchingResult:
    """Compute compatibility between a CV and a job offer."""
    user_prompt = _build_user_prompt(cv, job)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            raw = chat(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=3000,
            )
            json_str = extract_json_from_text(raw)
            data = json.loads(json_str)
            return MatchingResult.model_validate(data)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            if attempt < max_retries:
                continue

    raise ValueError(f"Failed to compute matching: {last_error}")