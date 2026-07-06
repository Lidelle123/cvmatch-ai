"""
PDF Extractor v2 — robust text extraction with word-level reconstruction.

Key improvement: uses extract_words() instead of extract_text() to handle
PDFs with poor character spacing (common with LaTeX-generated CVs).
"""
import re
from pathlib import Path
from typing import Optional

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails or yields unusable output."""
    pass


# ─────────────────────────────────────────────────────────────
# CORE EXTRACTION — word-level for better reliability
# ─────────────────────────────────────────────────────────────

def _group_words_into_lines(words: list[dict], y_tolerance: float = 3.0) -> list[list[dict]]:
    """
    Group words that are on the same horizontal line (same y-coordinate ± tolerance).
    Returns a list of lines, each line being a list of word dicts.
    """
    if not words:
        return []
    
    # Sort by vertical position (top), then by horizontal (x0)
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]
    current_y = sorted_words[0]["top"]
    
    for word in sorted_words[1:]:
        if abs(word["top"] - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # Sort current line by x position (left to right) before saving
            current_line.sort(key=lambda w: w["x0"])
            lines.append(current_line)
            current_line = [word]
            current_y = word["top"]
    
    if current_line:
        current_line.sort(key=lambda w: w["x0"])
        lines.append(current_line)
    
    return lines


def extract_raw_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF using word-level extraction.
    More robust than extract_text() for PDFs with kerning issues (LaTeX, etc.).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise PDFExtractionError(f"Not a PDF file: {pdf_path}")
    
    text_parts: list[str] = []
    
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) == 0:
                raise PDFExtractionError("PDF has no pages")
            
            for page in pdf.pages:
                # Extract words with positions — much more reliable
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,  # Use spatial positions, not flow
                )
                
                if not words:
                    continue
                
                # Group words by lines (based on vertical position)
                lines = _group_words_into_lines(words, y_tolerance=3)
                
                # Join words in each line with spaces, lines with newlines
                page_text = "\n".join(
                    " ".join(word["text"] for word in line)
                    for line in lines
                )
                text_parts.append(page_text)
    
    except Exception as e:
        if isinstance(e, (FileNotFoundError, PDFExtractionError)):
            raise
        raise PDFExtractionError(f"Failed to extract PDF: {e}")
    
    full_text = "\n\n".join(text_parts)
    
    if not full_text.strip():
        raise PDFExtractionError(
            "No text extracted. PDF might be image-based (scanned) — OCR required."
        )
    
    return full_text

def _split_concatenated_words(text: str) -> str:
    """
    Detect and split concatenated words like 'AlternanceDevData' → 'Alternance Dev Data'.
    
    Strategy: insert a space before a capital letter that follows a lowercase letter,
    UNLESS it's a common acronym pattern (AI, DAX, SQL, etc.) or a roman numeral.
    
    This is a heuristic — not perfect but catches 90% of LaTeX PDF concatenations.
    """
    # Step 1: split CamelCase patterns: lowercase+Uppercase
    # 'AlternanceDevData' → 'Alternance Dev Data'
    text = re.sub(r"([a-zà-ÿ])([A-ZÀ-Ÿ])", r"\1 \2", text)
    
    # Step 2: split letter+digit patterns: 'Master1' → 'Master 1' (only when clear)
    # We're conservative here to avoid breaking real concatenations like 'M1', 'M2', 'B1'
    # → Skip this step, too risky
    
    # Step 3: add space after punctuation if missing: 'AI|Analyse' → 'AI | Analyse'
    text = re.sub(r"([)\]|/,;:])([A-Za-zÀ-ÿ])", r"\1 \2", text)
    text = re.sub(r"([A-Za-zÀ-ÿ])([(\[])", r"\1 \2", text)
    
    # Step 4: split 'word(word' patterns: '(UniversitédeDschang' → '( Université de Dschang'
    text = re.sub(r"\(([A-ZÀ-Ÿ])", r"( \1", text)
    
    # Step 5: handle apostrophes glued: "d'unMaster" → "d'un Master"
    text = re.sub(r"(['']\w+?)([A-ZÀ-Ÿ])", r"\1 \2", text)
    
    return text


# ─────────────────────────────────────────────────────────────
# CLEANING — enhanced to handle real-world PDF artifacts
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text to make it LLM-ready.
    Now handles (cid:XXX) artifacts from unmapped glyphs.
    """
    # 🆕 Remove (cid:XXX) artifacts (unmapped glyphs, common with LaTeX PDFs)
    text = re.sub(r"\(cid:\d+\)", "", text)
    
    # 🆕 Split concatenated words BEFORE other cleaning
    text = _split_concatenated_words(text)
    
    # 🆕 Remove orphan ligature fragments left after cid removal
    # Patterns like " fl " (when ligature was on an icon) become noise
    # Only remove if surrounded by spaces (preserve real words)
    text = re.sub(r"\s+(fl|fi|ffi|ffl)\s+", " ", text)
    
    # Fix common ligatures (unicode form)
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬀ": "ff", "ﬅ": "st", "ﬆ": "st",
    }
    for old, new in ligatures.items():
        text = text.replace(old, new)
    
    # Normalize bullets
    for bullet in ["•", "●", "▪", "▫", "■", "□", "◦", "·", "½", "Ó", "¯"]:
        text = text.replace(bullet, "-")
    
    # Remove control chars except newline/tab
    text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Clean each line: collapse spaces, strip
    cleaned_lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# ─────────────────────────────────────────────────────────────
# QUALITY CHECKS — now checks SEMANTIC quality too
# ─────────────────────────────────────────────────────────────

def assess_extraction_quality(text: str) -> dict:
    """Heuristics to detect bad extractions, including semantic issues."""
    warnings: list[str] = []
    
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())
    
    if word_count < 50:
        warnings.append(f"Very few words ({word_count}). PDF might be scanned.")
    
    if char_count > 0:
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / char_count
        if alpha_ratio < 0.85:
            warnings.append(f"Low alphanumeric ratio ({alpha_ratio:.0%}). Garbled output?")
    
    # 🆕 Check for (cid:XXX) leftovers (should be cleaned but just in case)
    if "(cid:" in text:
        warnings.append("Unmapped glyphs detected (cid:XXX). Some chars couldn't be decoded.")
    
    # 🆕 Check for word concatenation (common LaTeX issue)
    # Heuristic: count words longer than 25 chars (suspiciously long)
    very_long_words = [w for w in text.split() if len(w) > 25]
    if len(very_long_words) > 3:
        warnings.append(
            f"{len(very_long_words)} very long words detected — possible word concatenation. "
            f"Example: '{very_long_words[0][:40]}...'"
        )
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "warnings": warnings,
        "is_usable": len(warnings) == 0,
    }


# ─────────────────────────────────────────────────────────────
# HIGH-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────

def extract_cv_text(pdf_path: str | Path, verbose: bool = False) -> str:
    """Full pipeline: extract → clean → quality check."""
    raw = extract_raw_text(pdf_path)
    cleaned = clean_text(raw)
    quality = assess_extraction_quality(cleaned)
    
    if verbose:
        print(f"📄 Extraction quality check:")
        print(f"   Characters : {quality['char_count']}")
        print(f"   Words      : {quality['word_count']}")
        print(f"   Lines      : {quality['line_count']}")
        if quality["warnings"]:
            print(f"   ⚠️  Warnings:")
            for w in quality["warnings"]:
                print(f"      - {w}")
        else:
            print(f"   ✅ Quality looks good")
    
    return cleaned