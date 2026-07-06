"""
Job offer parser — extracts structured data from a job offer text using a LLM.

Mirrors the architecture of cv_parser.py but for the supply side of recruitment:
takes a free-form job description and produces a typed, validated JobOffer object.
"""
import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

from src.llm_client import chat
from src.cv_parser import extract_json_from_text


# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────

class JobOffer(BaseModel):
    """Structured representation of a job offer."""
    job_title: str = Field(description="Title of the position")
    company: Optional[str] = Field(default=None, description="Hiring company")
    location: Optional[str] = Field(default=None, description="Job location")
    contract_type: Optional[str] = Field(
        default=None,
        description="CDI, CDD, Stage, Alternance, Freelance, etc."
    )
    seniority_level: Optional[str] = Field(
        default=None,
        description="Junior, Mid-level, Senior, etc."
    )
    summary: Optional[str] = Field(
        default=None,
        description="1-2 sentence role summary"
    )

    required_skills: list[str] = Field(
        default_factory=list,
        description="Mandatory technical and soft skills"
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Preferred but optional skills"
    )

    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(
        default_factory=list,
        description="Years of experience, education, certifications"
    )

    languages_required: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en analyse d'offres d'emploi.
Ton rôle est d'extraire les informations d'une offre d'emploi de manière structurée.

RÈGLES IMPÉRATIVES :
1. Tu réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après.
2. Tu respectes EXACTEMENT le schéma fourni.
3. Si une information est absente, mets `null` (champ) ou `[]` (liste).
4. Tu n'inventes JAMAIS d'informations.
5. Distingue bien les compétences REQUISES (must-have) des compétences
   SOUHAITÉES (nice-to-have / 'serait un plus' / 'bonus').
6. Préserve la langue originale de l'offre.
"""


def _build_user_prompt(job_text: str) -> str:
    schema = JobOffer.model_json_schema()
    return f"""Voici une offre d'emploi à analyser :

--- DÉBUT DE L'OFFRE ---
{job_text}
--- FIN DE L'OFFRE ---

Extrais les informations en JSON, en respectant EXACTEMENT ce schéma :

{json.dumps(schema, indent=2, ensure_ascii=False)}

Réponds UNIQUEMENT avec le JSON, sans markdown, sans ```json, sans commentaire.
"""


# ─────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────

def parse_job_offer(job_text: str, max_retries: int = 2, debug: bool = True) -> JobOffer:
    """Parse a raw job offer text into a structured JobOffer object."""
    user_prompt = _build_user_prompt(job_text)
    last_error = None
    last_raw_response = None

    for attempt in range(max_retries + 1):
        try:
            raw = chat(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2000,
            )
            last_raw_response = raw
            json_str = extract_json_from_text(raw)
            data = json.loads(json_str)
            return JobOffer.model_validate(data)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            print(f"⚠️  [job_parser] Attempt {attempt + 1} failed: {type(e).__name__}")
            if debug and last_raw_response:
                print(f"   🔍 Raw LLM response (first 500 chars):")
                print(f"   ---\n{last_raw_response[:500]}\n   ---")
            if attempt < max_retries:
                continue

    error_msg = f"Failed to parse job offer after {max_retries + 1} attempts: {last_error}"
    if last_raw_response:
        error_msg += f"\nLast LLM response (first 300 chars): {last_raw_response[:300]}"
    raise ValueError(error_msg)