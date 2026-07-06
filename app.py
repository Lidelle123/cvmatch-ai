"""
CVMatch AI — Streamlit web interface.

Run locally:    streamlit run app.py
Deploy:         push to GitHub → connect on share.streamlit.io
"""
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Bridge st.secrets to os.environ for Streamlit Cloud deployment
if "GROQ_API_KEY" not in os.environ:
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from src.pdf_extractor import extract_cv_text, PDFExtractionError
from src.cv_parser import parse_cv
from src.job_parser import parse_job_offer
from src.matcher import match_cv_to_job
from src.optimizer import (
    generate_optimized_cv,
    generate_cover_letter,
    generate_interview_questions,
)


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CVMatch AI",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 CVMatch AI")
st.caption(
    "Optimisez votre candidature avec l'IA — analyse, score, CV optimisé, "
    "lettre de motivation et préparation entretien."
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📥 Vos données")

    uploaded_cv = st.file_uploader(
        "1️⃣ Uploade ton CV (PDF)",
        type=["pdf"],
        help="CV au format PDF text-based (pas un scan)",
    )

    job_text = st.text_area(
        "2️⃣ Colle l'offre d'emploi",
        height=300,
        placeholder="Colle ici le texte complet de l'offre...",
    )

    analyze_button = st.button("🚀 Analyser", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("Powered by Groq · openai/gpt-oss-120b")
    st.caption("Built by [Vanella Lidelle Dzikang](https://github.com/Lidelle123)")


# ─────────────────────────────────────────────────────────────
# GUARDS
# ─────────────────────────────────────────────────────────────

if not analyze_button and "match" not in st.session_state:
    st.info(
        "👈 Uploade ton CV et colle une offre d'emploi dans la barre latérale, "
        "puis clique sur **Analyser**."
    )
    st.stop()

if analyze_button:
    # Reset state on new analysis
    for key in ["cv_data", "job_offer", "match", "optimized_cv", "cover_letter", "interview_prep"]:
        st.session_state.pop(key, None)

    if not uploaded_cv:
        st.error("❌ Aucun CV uploadé.")
        st.stop()

    if not job_text or len(job_text.strip()) < 100:
        st.error("❌ Le texte de l'offre est trop court (minimum 100 caractères).")
        st.stop()

    # ── PIPELINE STAGES ────────────────────────────────────────

    try:
        # Stage 1: extract CV text
        with st.spinner("📄 Extraction du CV..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_cv.getvalue())
                tmp_path = tmp.name
            try:
                cv_text = extract_cv_text(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Stage 2: parse CV
        with st.spinner("🧠 Analyse structurelle du CV..."):
            st.session_state.cv_data = parse_cv(cv_text)

        # Stage 3: parse job offer
        with st.spinner("🧠 Analyse de l'offre d'emploi..."):
            st.session_state.job_offer = parse_job_offer(job_text)

        # Stage 4: matching
        with st.spinner("⚖️  Calcul de la compatibilité..."):
            st.session_state.match = match_cv_to_job(
                st.session_state.cv_data,
                st.session_state.job_offer,
            )

    except (PDFExtractionError, FileNotFoundError) as e:
        st.error(f"❌ Erreur d'extraction PDF : {e}")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Erreur de traitement : {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {type(e).__name__} — {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────

cv_data = st.session_state.cv_data
job_offer = st.session_state.job_offer
match = st.session_state.match


# Score banner
verdict_labels = {
    "strong_match": "🟢 Excellent match",
    "good_match": "🟢 Bon match",
    "partial_match": "🟡 Match partiel",
    "weak_match": "🔴 Match faible",
}

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.metric(
        label="Score de compatibilité",
        value=f"{match.overall_score}/100",
        delta=verdict_labels.get(match.verdict, ""),
    )
    st.progress(match.overall_score / 100)


# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Analyse",
    "📄 CV optimisé",
    "✉️  Lettre de motivation",
    "🎤 Questions d'entretien",
    "🔍 Données extraites",
])


# ── Tab 1: analysis ──────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("✅ Compétences alignées")
        if match.skills_matched:
            for s in match.skills_matched:
                st.markdown(f"- {s}")
        else:
            st.caption("_(aucune compétence alignée détectée)_")

        if match.skills_bonus:
            st.subheader("⭐ Compétences bonus (nice-to-have)")
            for s in match.skills_bonus:
                st.markdown(f"- {s}")

    with col_b:
        st.subheader("❌ Compétences manquantes")
        if match.skills_missing:
            for s in match.skills_missing:
                st.markdown(f"- {s}")
        else:
            st.caption("_(aucun gap détecté)_")

    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("💪 Forces")
        for s in match.strengths:
            st.markdown(f"- {s}")
    with col_d:
        st.subheader("⚠️ Faiblesses")
        for w in match.weaknesses:
            st.markdown(f"- {w}")

    st.divider()
    st.subheader("📝 Analyse de l'expérience")
    st.write(match.experience_assessment)

    st.subheader("🎯 Recommandations")
    for rec in match.recommendations:
        st.markdown(f"- {rec}")


# ── Tab 2: optimized CV ─────────────────────────────────────
with tab2:
    if "optimized_cv" not in st.session_state:
        with st.spinner("✍️  Génération du CV optimisé..."):
            st.session_state.optimized_cv = generate_optimized_cv(cv_data, job_offer)

    st.markdown(st.session_state.optimized_cv)
    st.download_button(
        "💾 Télécharger en Markdown",
        st.session_state.optimized_cv,
        file_name=f"cv_optimise_{job_offer.job_title.replace(' ', '_')[:30]}.md",
        mime="text/markdown",
    )


# ── Tab 3: cover letter ─────────────────────────────────────
with tab3:
    if "cover_letter" not in st.session_state:
        with st.spinner("✍️  Génération de la lettre de motivation..."):
            st.session_state.cover_letter = generate_cover_letter(cv_data, job_offer)

    st.markdown(st.session_state.cover_letter)
    st.download_button(
        "💾 Télécharger en TXT",
        st.session_state.cover_letter,
        file_name=f"lettre_motivation_{job_offer.job_title.replace(' ', '_')[:30]}.txt",
        mime="text/plain",
    )


# ── Tab 4: interview questions ──────────────────────────────
with tab4:
    if "interview_prep" not in st.session_state:
        with st.spinner("🎤 Génération des questions d'entretien..."):
            st.session_state.interview_prep = generate_interview_questions(cv_data, job_offer)

    category_emojis = {
        "technical": "🛠️",
        "behavioral": "🧠",
        "experience": "💼",
        "motivation": "🔥",
        "company_specific": "🏢",
    }

    for i, q in enumerate(st.session_state.interview_prep.questions, 1):
        emoji = category_emojis.get(q.category, "❓")
        with st.expander(f"{emoji} Q{i} — {q.question}"):
            st.markdown("**Pourquoi cette question ?**")
            st.write(q.why_asked)
            st.markdown("**Stratégie de réponse :**")
            st.write(q.answer_strategy)


# ── Tab 5: raw extracted data ───────────────────────────────
with tab5:
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("📄 CV structuré")
        st.json(cv_data.model_dump())
    with col_y:
        st.subheader("📋 Offre structurée")
        st.json(job_offer.model_dump())