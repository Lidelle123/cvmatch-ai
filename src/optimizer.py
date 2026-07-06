"""
Optimizer — generates tailored deliverables for a specific application.

Three features (one per function), each backed by a specialized prompt:
- generate_optimized_cv     : rewrites the CV to align with the offer
- generate_cover_letter     : produces a personalized cover letter
- generate_interview_questions : predicts likely interview questions
"""
import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from src.llm_client import chat
from src.cv_parser import CVData, extract_json_from_text
from src.job_parser import JobOffer


# ─────────────────────────────────────────────────────────────
# 1. OPTIMIZED CV
# ─────────────────────────────────────────────────────────────

OPTIMIZED_CV_PROMPT = """Tu es un expert en rédaction de CV pour le recrutement tech.
Ton rôle : reformuler un CV pour qu'il soit OPTIMISÉ pour une offre d'emploi spécifique.

RÈGLES :
1. Tu n'inventes JAMAIS d'expérience ou compétence absente du CV original.
2. Tu RÉORGANISES et REFORMULES pour faire ressortir ce qui est pertinent.
3. Tu utilises les mots-clés EXACTS de l'offre quand le candidat a la compétence (ATS-friendly).
4. Format MARKDOWN structuré : sections claires, bullets percutants.
5. Chaque bullet d'expérience commence par un verbe d'action fort (Conçu, Développé, etc.).
6. Quantifie quand possible (X%, Y utilisateurs) — sans inventer de chiffres.
7. Préserve la langue originale du CV.
"""


def generate_optimized_cv(cv: CVData, job: JobOffer) -> str:
    """Generate a CV rewritten in Markdown, optimized for the target job offer."""
    user_prompt = f"""CV original :
{cv.model_dump_json(indent=2)}

Offre visée :
{job.model_dump_json(indent=2)}

Réécris le CV en MARKDOWN, optimisé pour cette offre.
Structure : Nom + titre + contact, Profil, Compétences, Expériences, Formation, Projets/Certifs.
"""
    return chat(
        prompt=user_prompt,
        system_prompt=OPTIMIZED_CV_PROMPT,
        temperature=0.4,
        max_tokens=3000,
    )


# ─────────────────────────────────────────────────────────────
# 2. COVER LETTER
# ─────────────────────────────────────────────────────────────

COVER_LETTER_PROMPT = """Tu es un coach en candidatures et expert en lettres de motivation.
Tu écris des lettres PERSONNALISÉES, sincères, et impactantes.

RÈGLES :
1. Tu n'inventes JAMAIS d'informations.
2. Ton professionnel mais HUMAIN, jamais robotique.
3. Pas de clichés ("dynamique et motivé", "passionné depuis toujours", etc.).
4. Structure en 3-4 paragraphes :
   - Accroche personnalisée (pourquoi cette entreprise, ce rôle)
   - Pourquoi je suis le bon profil (2-3 points forts liés à l'offre)
   - Ce que je veux apporter et apprendre
   - Conclusion ouvrant sur un échange
5. Maximum 350 mots.
6. Tu écris dans la langue de l'offre.
"""


def generate_cover_letter(cv: CVData, job: JobOffer) -> str:
    """Generate a personalized cover letter."""
    user_prompt = f"""CV :
{cv.model_dump_json(indent=2)}

Offre :
{job.model_dump_json(indent=2)}

Rédige une lettre de motivation personnalisée et impactante.
"""
    return chat(
        prompt=user_prompt,
        system_prompt=COVER_LETTER_PROMPT,
        temperature=0.7,
        max_tokens=1500,
    )


# ─────────────────────────────────────────────────────────────
# 3. INTERVIEW QUESTIONS
# ─────────────────────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    category: Literal["technical", "behavioral", "experience", "motivation", "company_specific"]
    question: str
    why_asked: str = Field(description="Why this question is likely to be asked")
    answer_strategy: str = Field(description="How to approach this question")


class InterviewPrep(BaseModel):
    questions: list[InterviewQuestion] = Field(min_length=8, max_length=12)


INTERVIEW_PROMPT = """Tu es un recruteur tech expérimenté et un coach d'entretien.
Tu prédis les questions susceptibles d'être posées en entretien, basé sur le CV et l'offre.

RÈGLES :
1. Génère ENTRE 8 et 12 questions au total, équilibrées par catégorie.
2. Mélange : technique, comportementale, expérience, motivation, et 1+ spécifique à
   l'entreprise/secteur.
3. Pour chaque question : POURQUOI elle sera posée + STRATÉGIE de réponse.
4. Les questions techniques ciblent les compétences clés de l'offre.
5. Les questions comportementales suggèrent le format STAR (Situation-Tâche-Action-Résultat).
6. Réponds UNIQUEMENT en JSON valide.
"""


def generate_interview_questions(cv: CVData, job: JobOffer, max_retries: int = 2) -> InterviewPrep:
    """Generate predicted interview questions with answer strategies."""
    schema = InterviewPrep.model_json_schema()
    user_prompt = f"""CV :
{cv.model_dump_json(indent=2)}

Offre :
{job.model_dump_json(indent=2)}

Génère les questions probables, avec ce schéma :
{json.dumps(schema, indent=2, ensure_ascii=False)}

Réponds UNIQUEMENT en JSON valide.
"""

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = chat(
                prompt=user_prompt,
                system_prompt=INTERVIEW_PROMPT,
                temperature=0.5,
                max_tokens=6000,
            )
            json_str = extract_json_from_text(raw)
            data = json.loads(json_str)
            return InterviewPrep.model_validate(data)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            if attempt < max_retries:
                continue

    raise ValueError(f"Failed to generate interview questions: {last_error}")