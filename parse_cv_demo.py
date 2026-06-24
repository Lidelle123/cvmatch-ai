"""
Demo — extract structured data from a sample CV.
Run: python parse_cv_demo.py
"""
from src.cv_parser import parse_cv

# Pour ce premier test, on utilise un CV en texte brut.
# Demain, on extraira ce texte depuis ton vrai CV en PDF.
SAMPLE_CV = """
Vanella Lidelle DZIKANG
AI Engineer in training | Étudiante en cybersécurité
Email : pas-un-email
Téléphone : +237 6XX XXX XXX
Localisation : Douala, Cameroun

PROFIL
Étudiante passionnée par les métiers de la data et de l'IA,
actuellement en formation à ECE Paris.

COMPÉTENCES
Python, Pandas, Scikit-learn, SQL, Streamlit, Git, Linux

EXPÉRIENCES
- Apprentie Cybersécurité chez Intuitem (2024-2025)
  Développement sur la plateforme CISO Assistant (GRC open source).

- Stagiaire Data chez Kekottech (2023, 6 mois)
  Analyse de données pour des PME camerounaises.

FORMATION
- Master en Cybersécurité, ECE Paris (en cours, 2026)
- Bachelor en Informatique, Université de Dschang (2022)

LANGUES
Français (natif), Anglais (courant)
"""


if __name__ == "__main__":
    print("🔍 Parsing the CV...\n")
    cv = parse_cv(SAMPLE_CV)
    
    print("=" * 60)
    print("✅ EXTRACTED DATA")
    print("=" * 60)
    print(f"👤 Name        : {cv.full_name}")
    print(f"📧 Email       : {cv.email}")
    print(f"📍 Location    : {cv.location}")
    print(f"💼 Title       : {cv.title}")
    print(f"\n🛠️  Skills ({len(cv.skills)}):")
    for skill in cv.skills:
        print(f"   - {skill}")
    
    print(f"\n💼 Experiences ({len(cv.experiences)}):")
    for exp in cv.experiences:
        print(f"   - {exp.job_title} @ {exp.company} ({exp.duration})")
    
    print(f"\n🎓 Education ({len(cv.education)}):")
    for edu in cv.education:
        print(f"   - {edu.degree} @ {edu.institution} ({edu.year})")
    
    print(f"\n🌍 Languages : {', '.join(cv.languages)}")
    
    # Bonus : afficher tout en JSON aussi
    print("\n" + "=" * 60)
    print("📄 FULL JSON")
    print("=" * 60)
    print(cv.model_dump_json(indent=2))