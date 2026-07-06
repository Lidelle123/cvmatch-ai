"""
Day 3 demo — Full pipeline: PDF → text → structured CV data.

Usage:
    python pdf_to_structured_demo.py <path-to-cv.pdf>
    python pdf_to_structured_demo.py data/cvs/my_cv.pdf
"""
import sys
from pathlib import Path

from src.pdf_extractor import extract_cv_text, PDFExtractionError
from src.cv_parser import parse_cv


def main(pdf_path: str) -> None:
    print("=" * 60)
    print(f"📂 Processing: {pdf_path}")
    print("=" * 60)
    
    # Stage 1: Extract & clean PDF text
    print("\n📄 Stage 1: PDF extraction...")
    try:
        cv_text = extract_cv_text(pdf_path, verbose=True)
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    except PDFExtractionError as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Show a preview
    preview = cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
    print(f"\n📝 Text preview:\n---\n{preview}\n---")
    
    # Stage 2: LLM parsing
    print("\n🤖 Stage 2: LLM structured extraction...")
    try:
        cv_data = parse_cv(cv_text, debug=True)
    except ValueError as e:
        print(f"❌ Parsing failed: {e}")
        sys.exit(1)
    
    # Stage 3: Display results
    print("\n" + "=" * 60)
    print("✅ STRUCTURED CV DATA")
    print("=" * 60)
    print(f"👤 {cv_data.full_name}")
    if cv_data.title:
        print(f"💼 {cv_data.title}")
    if cv_data.email:
        print(f"📧 {cv_data.email}")
    if cv_data.location:
        print(f"📍 {cv_data.location}")
    
    if cv_data.skills:
        print(f"\n🛠️  Skills ({len(cv_data.skills)}):")
        for s in cv_data.skills:
            print(f"   - {s}")
    
    if cv_data.experiences:
        print(f"\n💼 Experiences ({len(cv_data.experiences)}):")
        for exp in cv_data.experiences:
            print(f"   - {exp.job_title} @ {exp.company}")
            if exp.duration:
                print(f"     {exp.duration}")
    
    if cv_data.education:
        print(f"\n🎓 Education ({len(cv_data.education)}):")
        for edu in cv_data.education:
            print(f"   - {edu.degree} @ {edu.institution} ({edu.year or 'N/A'})")
    
    # Save the JSON output
    output_path = Path("data/outputs") / (Path(pdf_path).stem + "_parsed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cv_data.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_structured_demo.py <path-to-cv.pdf>")
        print("Example: python pdf_to_structured_demo.py data/cvs/my_cv.pdf")
        sys.exit(1)
    
    main(sys.argv[1])