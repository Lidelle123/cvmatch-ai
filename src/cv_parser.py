"""
CV Parser — extracts structured data from raw CV text using a LLM.

Concepts demonstrated:
- Pydantic models for schema definition
- LLM prompting for JSON output
- Validation & error handling
- Retry on malformed output
"""
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

from src.llm_client import chat


# ─────────────────────────────────────────────────────────────
# 1. SCHEMA — what we expect from the LLM
# ─────────────────────────────────────────────────────────────

class Experience(BaseModel):
    """A single professional experience."""
    job_title: str = Field(description="Job title or role")
    company: str = Field(description="Company or organization name")
    duration: Optional[str] = Field(
        default=None,
        description="Duration like '2 years' or '2022-2024'"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief summary of responsibilities and achievements"
    )


class Education(BaseModel):
    """A single education entry."""
    degree: str = Field(description="Diploma or degree name")
    institution: str = Field(description="School or university name")
    year: Optional[str] = Field(default=None, description="Year or year range")


class CVData(BaseModel):
    """Structured representation of a CV."""
    full_name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City and/or country")
    
    title: Optional[str] = Field(
        default=None,
        description="Current professional title or headline"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief professional summary / about me"
    )
    
    skills: list[str] = Field(
        default_factory=list,
        description="List of technical and soft skills"
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Spoken languages"
    )
    
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# 2. PROMPT — how we instruct the LLM
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en analyse de CV.
Ton rôle est d'extraire les informations d'un CV de manière structurée.

RÈGLES IMPÉRATIVES :
1. Tu réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après.
2. Tu respectes EXACTEMENT le schéma fourni.
3. Si une information est absente du CV, mets `null` (champ) ou `[]` (liste).
4. Tu n'inventes JAMAIS d'informations qui ne sont pas dans le CV.
5. Tu préserves la langue originale du CV.

ATTENTION — pièges courants à éviter :
- Les titres de sections (PROFIL, EXPÉRIENCES, FORMATION, COMPÉTENCES, PROJETS,
  CERTIFICATIONS, LANGUES, ATOUTS, TECHNOLOGIES, etc.) ne sont JAMAIS des noms
  d'entreprise ou de poste. Ignore-les complètement.
- Le nom de l'entreprise vient JUSTE après le titre du poste dans une expérience.
- Si tu vois des mots collés sans espace (ex: 'DiplôméedunMaster'), reconstitue
  le sens et reformule proprement.
- Si une date est collée (ex: '2025-Mars2026'), reformate proprement.
"""


def build_user_prompt(cv_text: str) -> str:
    """Build the user prompt with CV text and schema."""
    schema = CVData.model_json_schema()
    return f"""Voici un CV à analyser :

--- DÉBUT DU CV ---
{cv_text}
--- FIN DU CV ---

Extrais les informations en JSON, en respectant EXACTEMENT ce schéma :

{json.dumps(schema, indent=2, ensure_ascii=False)}

Réponds UNIQUEMENT avec le JSON, sans markdown, sans ```json, sans commentaire.
"""


# ─────────────────────────────────────────────────────────────
# 3. UTILS — clean LLM output
# ─────────────────────────────────────────────────────────────

def extract_json_from_text(text: str) -> str:
    """
    LLMs sometimes wrap JSON in markdown code blocks or add text around it.
    This function extracts the JSON object from such text.
    """
    # Strip markdown code blocks if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    
    # Find first { and last } to extract the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    
    return text[start:end + 1]


# ─────────────────────────────────────────────────────────────
# 4. CORE FUNCTION
# ─────────────────────────────────────────────────────────────

def parse_cv(cv_text: str, max_retries: int = 2, debug: bool = False) -> CVData:
    """
    Parse a raw CV text into a structured CVData object.
    
    Args:
        cv_text: The raw text of the CV.
        max_retries: Retries on malformed output.
        debug: If True, prints the raw LLM response when parsing fails.
    """
    user_prompt = build_user_prompt(cv_text)
    last_error = None
    last_raw_response = None
    
    for attempt in range(max_retries + 1):
        try:
            raw_response = chat(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2000,
            )
            last_raw_response = raw_response
            
            json_str = extract_json_from_text(raw_response)
            data = json.loads(json_str)
            cv_data = CVData.model_validate(data)
            return cv_data
        
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            print(f"⚠️  Attempt {attempt + 1} failed: {type(e).__name__}")
            
            if debug and last_raw_response:
                print(f"   🔍 Raw LLM response (first 500 chars):")
                print(f"   ---")
                print(f"   {last_raw_response[:500]}")
                print(f"   ---")
            
            if attempt < max_retries:
                print(f"   Retrying...")
                continue
    
    # Final error message includes raw response for diagnostic
    error_msg = f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
    if last_raw_response:
        error_msg += f"\nLast LLM response (first 300 chars): {last_raw_response[:300]}"
    raise ValueError(error_msg)