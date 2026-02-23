"""
Document Parsing Test Script (v4)
==================================
Dual section detection: text-regex + font-analysis, merged and deduplicated.

Usage:
    python document_parsing.py <pdf_path> [--output report.txt] [--skip-vision] [--skip-ocr]
"""

import os, io, re, sys, argparse, warnings
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import fitz
except ImportError:
    sys.exit("ERROR: pip install pymupdf")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import tabula
except ImportError:
    tabula = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    PDF_PATH: str = ""
    OUTPUT_PATH: str = "parsing_report.txt"
    SKIP_VISION: bool = False
    SKIP_OCR: bool = False
    SEC_CHUNK_SIZE: int = 800
    SEC_CHUNK_OVERLAP: int = 200
    TABLE_CHUNK_SIZE: int = 1400
    TABLE_CHUNK_OVERLAP: int = 50
    FIGURE_CHUNK_SIZE: int = 500
    FIGURE_CHUNK_OVERLAP: int = 80
    REF_CHUNK_SIZE: int = 700
    REF_CHUNK_OVERLAP: int = 50
    GOOGLE_API_KEY: str = ""


# ============================================================================
# REPORT WRITER
# ============================================================================

class ReportWriter:
    def __init__(self):
        self.sections = []

    def header(self, t):
        self.sections.append(f"\n{'='*80}\n{t}\n{'='*80}\n")
        print(f"\n{'='*60}\n{t}\n{'='*60}")

    def subheader(self, t):
        self.sections.append(f"\n{'─'*60}\n{t}\n{'─'*60}\n")
        print(f"  ── {t}")

    def line(self, t=""):
        self.sections.append(t)

    def kv(self, k, v, indent=2):
        self.sections.append(f"{' '*indent}{k}: {v}")

    def preview(self, t, mx=500, indent=4):
        c = t.replace("\n", " ").strip()
        p = c[:mx] + (" [...]" if len(c) > mx else "")
        self.sections.append(f"{' '*indent}{p}")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.sections))
        print(f"\n✅ Report saved to: {path}")


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text):
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"\f", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(nan|NaN)\b", "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================================
# SHARED PATTERNS
# ============================================================================

_TITLE_REJECT = re.compile(
    r"(?i)(arxiv[:\s]|preprint|submitted to|accepted at|under review|"
    r"\d{4}\.\d{4,}|cs\.\w{2}|stat\.\w{2}|math\.\w{2}|"
    r"\[\w+\.\w+\]|\bv\d+\b|\d{1,2}\s+(jan|feb|mar|apr|may|jun|"
    r"jul|aug|sep|oct|nov|dec)\w*\s+\d{4})"
)

_ORG_REJECT = re.compile(
    r"(?i)^(university|institute|department|school|college|center|"
    r"laboratory|lab\b|research|foundation|independent|"
    r"google|microsoft|meta\b|openai|deepmind|facebook|amazon|"
    r"ibm|nvidia|rutgers|stanford|mit\b|cmu\b|aios|"
    r"anthropic|apple|intel|huawei|tencent|bytedance)"
)

_ORG_KEYWORDS = [
    r"University", r"Institute", r"Laboratory", r"Lab\b",
    r"Department", r"School", r"College", r"Center",
    r"Corporation", r"Inc\.", r"Ltd\.", r"Research",
    r"Foundation",
    r"Google", r"Microsoft", r"Meta\b", r"OpenAI", r"DeepMind",
    r"Facebook", r"Amazon", r"IBM", r"NVIDIA",
    r"Anthropic", r"Apple", r"Intel", r"Huawei", r"Tencent",
    r"ByteDance", r"Samsung", r"Alibaba",
]

_EXCLUDE_HEADINGS = {"references", "bibliography", "reference"}

_KEYWORD_HEADINGS = {
    "abstract", "acknowledgements", "acknowledgments",
    "appendix", "appendices",
}


# ============================================================================
# 1. TITLE
# ============================================================================

def extract_title(doc, header_text):
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    candidates = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if span["size"] > 14 and len(text) > 10:
                        if not _TITLE_REJECT.search(text):
                            candidates.append((span["size"], text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if candidates:
        title = candidates[0][1]
        top = candidates[0][0]
        for sz, txt in candidates[1:]:
            if abs(sz - top) < 1.0 and len(title) < 150:
                title += " " + txt
            else:
                break
        return clean_text(title)
    for ln in header_text.split("\n")[:15]:
        ln = ln.strip()
        if len(ln) > 20 and not _TITLE_REJECT.search(ln):
            if not re.match(r"^(Abstract|Introduction|\d+\.)", ln, re.IGNORECASE):
                return clean_text(ln)
    return ""


# ============================================================================
# 2. AUTHORS
# ============================================================================

def extract_authors(header_text, title=""):
    authors = []
    title_lower = title.lower().strip() if title else ""
    for line in header_text.split("\n")[:30]:
        line = line.strip()
        if not line or len(line) > 200:
            continue
        if re.match(r"^(Abstract|Introduction|\d+\.|Keywords|arXiv)", line, re.IGNORECASE):
            continue
        if "@" in line or _TITLE_REJECT.search(line):
            continue
        if title_lower and line.lower().strip() == title_lower:
            continue
        cleaned = re.sub(r"[∗†‡§¶]+", "", line)
        cleaned = re.sub(r"(?<=[a-zA-Z])\d{1,2}(?=[,\s]|$)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not re.search(r"[A-ZÀ-ÖØ-ÞĀ-ŽÇ][a-zà-öø-ÿā-žç]+\s+[A-ZÀ-ÖØ-ÞĀ-ŽÇ]", cleaned):
            continue
        words = cleaned.split()
        cap = sum(1 for w in words if w and (w[0].isupper() or ord(w[0]) > 127)) / max(len(words), 1)
        if cap < 0.5:
            continue
        for name in re.split(r"\s+and\s+|,\s*", cleaned):
            name = name.strip()
            if not name or _ORG_REJECT.search(name):
                continue
            if title_lower and name.lower() == title_lower:
                continue
            parts = name.split()
            if 2 <= len(parts) <= 5:
                if all(p[0].isupper() or ord(p[0]) > 127 for p in parts if p):
                    if name not in authors:
                        authors.append(name)
    return authors[:20]


# ============================================================================
# 3. EMAILS
# ============================================================================

def extract_emails(header_text):
    return list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", header_text)))


# ============================================================================
# 4. ORGANIZATIONS
# ============================================================================

def extract_organizations(header_text):
    orgs = []
    for line in header_text.split("\n")[:40]:
        line = line.strip()
        if re.match(r"^[a-zA-Z0-9._%+-]+@", line):
            continue
        if len(line) > 120 or (line.endswith(".") and line.count(" ") > 8):
            continue
        for kw in _ORG_KEYWORDS:
            if re.search(kw, line, re.IGNORECASE):
                cleaned = re.sub(r"[∗†‡§¶\d]+", "", line).strip()
                if len(cleaned) > 5 and cleaned not in orgs:
                    orgs.append(cleaned)
                break
    return orgs[:10]


# ============================================================================
# 5. ABSTRACT
# ============================================================================

def extract_abstract(header_text):
    for pat in [
        r"Abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1\s+[A-Z]|\bIntroduction\b|Keywords)",
        r"ABSTRACT\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1\s+[A-Z]|\bINTRODUCTION\b|Keywords)",
    ]:
        m = re.search(pat, header_text, re.DOTALL | re.IGNORECASE)
        if m:
            abstract = re.sub(r"\s+", " ", m.group(1).strip())
            if len(abstract) > 50:
                return abstract[:2000]
    return ""


# ============================================================================
# 6. SECTION DETECTION — METHOD A: FONT ANALYSIS (NEW)
# ============================================================================

def _extract_sections_by_font(doc):
    """
    Extract section headings by analyzing font properties (size, bold).
    Scientific papers use distinct fonts for headings vs body text.
    
    Strategy:
    1. Collect all text spans with their font size, weight, and position
    2. Determine the body text font size (most common size)
    3. Find spans that are larger or bolder than body text
    4. Filter those to ones that look like section headings
    """
    # Collect all spans across all pages
    all_spans = []
    for pn in range(len(doc)):
        page = doc[pn]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_spans = []
                for span in line["spans"]:
                    line_spans.append({
                        "text": span["text"],
                        "size": round(span["size"], 1),
                        "font": span["font"],
                        "bold": "bold" in span["font"].lower() or "Bold" in span["font"],
                        "page": pn,
                        "y": round(line["bbox"][1], 1),
                    })
                if line_spans:
                    # Merge spans in same line
                    combined_text = " ".join(s["text"] for s in line_spans).strip()
                    if combined_text:
                        all_spans.append({
                            "text": combined_text,
                            "size": max(s["size"] for s in line_spans),
                            "bold": any(s["bold"] for s in line_spans),
                            "font": line_spans[0]["font"],
                            "page": line_spans[0]["page"],
                            "y": line_spans[0]["y"],
                        })

    if not all_spans:
        return []

    # Determine body text size (most frequent font size)
    from collections import Counter
    size_counts = Counter()
    for s in all_spans:
        if len(s["text"]) > 20:  # only count substantial text
            size_counts[s["size"]] += len(s["text"])
    
    if not size_counts:
        return []
    
    body_size = size_counts.most_common(1)[0][0]

    # Find heading candidates: bold or larger than body text
    heading_candidates = []
    for s in all_spans:
        text = s["text"].strip()
        if not text or len(text) < 3:
            continue
        
        is_heading_font = False
        
        # Larger than body text
        if s["size"] > body_size + 0.5:
            is_heading_font = True
        
        # Bold at body size (common for subsection headings)
        if s["bold"] and abs(s["size"] - body_size) < 1.5:
            is_heading_font = True
        
        if not is_heading_font:
            continue
        
        # Filter: must look like a heading, not random large text
        # Skip very long text (body paragraphs in bold)
        if len(text) > 150:
            continue
        
        # Skip if it looks like a title (too large, on first page top)
        if s["size"] > body_size + 4 and s["page"] == 0:
            continue
        
        # Skip email, URL, arXiv lines
        if "@" in text or "http" in text.lower() or _TITLE_REJECT.search(text):
            continue

        heading_candidates.append({
            "text": text,
            "size": s["size"],
            "bold": s["bold"],
            "page": s["page"],
            "y": s["y"],
        })

    # Now parse heading candidates into structured sections
    sections = []
    for h in heading_candidates:
        text = h["text"].strip()
        
        # Try to match numbered heading: "1 Introduction", "3.2.1 Scaled..."
        m = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", text)
        if not m:
            m = re.match(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", text)
        if not m:
            # Appendix: "A Experiment", "B.1 Prompt Template"
            m = re.match(r"^([A-Z](?:\.\d+)*)\s+(.+)$", text)
        
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            # Validate section number
            if re.match(r"^\d+(?:\.\d+)*$", num):
                parts = num.split(".")
                if len(parts[0]) > 2 or len(parts) > 4:
                    continue
                if any(len(p) > 2 for p in parts[1:]):
                    continue
                # Reject decimal-like numbers (sub-parts > 20)
                for p in parts[1:]:
                    if int(p) > 20:
                        num = None
                        break
                if num is None:
                    continue
                # Top-level too high
                if len(parts) == 1 and int(parts[0]) > 20:
                    continue
            elif re.match(r"^[A-Z](?:\.\d+)*$", num):
                if len(num.split(".")) > 4:
                    continue
            else:
                continue
            
            if len(title) < 2 or len(title) > 120:
                continue
            
            sections.append({
                "heading": text,
                "number": num,
                "title": title,
                "page": h["page"],
                "y": h["y"],
                "pattern": "font_numbered",
                "size": h["size"],
                "bold": h["bold"],
            })
        else:
            # Unnumbered heading (e.g., "Abstract", "Related Work")
            # Must be short and capitalized
            if len(text) > 80:
                continue
            if not text[0].isupper():
                continue
            # Skip figure/table captions — handled separately
            if re.match(r"^(Table|Figure)\s+\d+", text, re.IGNORECASE):
                continue
            # Skip obvious non-headings
            if text.lower() in ("preprint", "preprint."):
                continue
            
            sections.append({
                "heading": text,
                "number": None,
                "title": text,
                "page": h["page"],
                "y": h["y"],
                "pattern": "font_unnumbered",
                "size": h["size"],
                "bold": h["bold"],
            })

    return sections


# ============================================================================
# 7. SECTION DETECTION — METHOD B: TEXT REGEX (ORIGINAL, IMPROVED)
# ============================================================================

def _get_normalized_lines(full_text):
    lines = []
    for ln in full_text.splitlines():
        s = re.sub(r"\s+", " ", (ln or "")).strip()
        if s:
            lines.append(s)
    return lines


def _extract_sections_by_text(full_text):
    """
    Section extraction from original scientific-rag notebook logic.
    Uses normalized text lines with split-header merging, ALL_CAPS detection,
    table caption detection, keyword headings, and numbered heading parsing.
    """

    # --- Exclusion sets from original notebook ---
    acronym_exclusions = {
        'BLEU', 'GPU', 'CPU', 'LSTM', 'RNN', 'CNN', 'NLP', 'WMT', 'TPU',
        'ADAM', 'SGD', 'RELU', 'GELU', 'FFN', 'MLP', 'BERT', 'GPT', 'LLM',
        'PPL', 'GNMT', 'API', 'URL', 'HTTP', 'HTML', 'JSON', 'XML', 'SQL',
        'ROUGE', 'METEOR', 'SBERT', 'RAM', 'ROM',
        'LOCOMO', 'READAGENT', 'MEMORYBANK', 'MEMGPT', 'AMEM',
        'LLAMA', 'QWEN', 'DEEPSEEK',
    }

    keyword_headings = {
        "abstract", "acknowledgements", "acknowledgments",
        "appendix", "appendices",
    }

    # --- Helper functions from original notebook ---
    def is_number_only(s):
        return re.match(r"^\d+(?:\.\d+)*\.?$", s) is not None

    def is_appendix_number_only(s):
        return re.match(r"^[A-Z](?:\.\d+)*\.?$", s) is not None

    def is_valid_section_number(num):
        if not re.match(r"^\d+(?:\.\d+)*$", num):
            return False
        parts = num.split(".")
        if len(parts) > 4:
            return False
        if len(parts[0]) > 2:
            return False
        if any(len(p) > 2 for p in parts[1:]):
            return False
        # Reject decimal-looking numbers: sub-parts > 9 are unlikely section numbers
        # Real sections: 3.1, 3.2.1, 4.7 — sub-parts are small
        # Decimals: 15.32, 18.02, 25.87 — sub-parts are large
        for p in parts[1:]:
            if int(p) > 20:
                return False
        # Top-level sections rarely go above 10-15 in a paper
        if len(parts) == 1 and int(parts[0]) > 20:
            return False
        return True

    def is_valid_appendix_number(num):
        if not re.match(r"^[A-Z](?:\.\d+)*$", num):
            return False
        return len(num.split(".")) <= 4

    def looks_like_title(title):
        if len(title) < 3 or len(title) > 90:
            return False
        if not title[0].isalpha():
            return False
        if title.endswith(".") and any(c.islower() for c in title):
            return False
        alpha = sum(c.isalpha() for c in title)
        if alpha < 0.65 * len(title):
            return False
        return True

    def is_reference_entry(s):
        return re.match(r"^\[\d+\]\s+", s) is not None

    def is_table_caption(s):
        return re.match(r"^(Table|TABLE|Figure|FIGURE)\s+\d+\s*[:\.]", s) is not None

    def next_line_is_data(idx):
        if idx + 1 >= len(lines):
            return False
        nxt = lines[idx + 1]
        if not nxt:
            return False
        # Skip URL/arxiv lines
        if re.search(r"https?://|arxiv|@", nxt, re.IGNORECASE):
            return False
        # Check if mostly digits
        alpha_count = sum(c.isalpha() for c in nxt)
        if alpha_count > 30:
            return False
        digit_ratio = sum(c.isdigit() or c in ".,·±−+% " for c in nxt) / max(len(nxt), 1)
        return digit_ratio > 0.5

    # --- Normalize lines (from original notebook) ---
    lines = _get_normalized_lines(full_text)

    sections = []
    in_references = False

    i = 0
    while i < len(lines):
        s = lines[i]

        # Skip reference entries once we hit references section
        if in_references:
            if is_reference_entry(s):
                i += 1
                continue
            i += 1
            continue

        # 1) Table/Figure captions (from original)
        if is_table_caption(s):
            m = re.match(r"^(?P<pre>Table|TABLE|Figure|FIGURE)\s+(?P<num>\d+)\s*[:.]\s*(?P<title>.+)$", s)
            if m:
                pre = m.group("pre").capitalize()
                sections.append({
                    "heading": s,
                    "line_number": i,
                    "number": f"{pre} {m.group('num')}",
                    "title": f"{pre} {m.group('num')}: {m.group('title').strip()}",
                    "pattern": "text_caption",
                })
            i += 1
            continue

        # 2) Merge split headers: "3.2.1" on one line + title on next (from original)
        if is_number_only(s) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if looks_like_title(nxt):
                s = f"{s.rstrip('.')} {nxt}"
                i += 1

        # 2b) Same for appendix numbers: "A" on one line + title on next
        if is_appendix_number_only(s) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if looks_like_title(nxt):
                s = f"{s.rstrip('.')} {nxt}"
                i += 1

        # 3) Skip reference entries (from original)
        if is_reference_entry(s):
            i += 1
            continue

        # 4) Keyword headings: Abstract, Acknowledgements, etc. (from original)
        if s.strip().lower() in keyword_headings:
            title = s.strip()
            sections.append({
                "heading": title,
                "line_number": i,
                "number": None,
                "title": title,
                "pattern": "text_keyword",
            })
            i += 1
            continue

        # 4b) References boundary — don't add as section, just set flag
        if s.strip().lower() in ("references", "bibliography", "reference"):
            in_references = True
            i += 1
            continue

        # 5) ALL CAPS headings (from original, with data-line guard)
        m_caps = re.match(r"^([A-Z]{3,}(?:\s+[A-Z]{3,})*)$", s)
        if m_caps:
            heading = m_caps.group(1)
            if heading not in acronym_exclusions and len(heading) >= 5:
                if not next_line_is_data(i):
                    sections.append({
                        "heading": heading,
                        "line_number": i,
                        "number": None,
                        "title": heading,
                        "pattern": "text_allcaps",
                    })
                    if heading.lower() == "references":
                        in_references = True
            i += 1
            continue

        # 6) Numbered headings (from original): "1 Introduction", "6.1 Machine Translation"
        m_num = re.match(r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+)$", s)
        if m_num:
            num = m_num.group("num")
            title = m_num.group("title").strip()
            if is_valid_section_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}",
                    "line_number": i,
                    "number": num,
                    "title": title,
                    "pattern": "text_numbered",
                })
                i += 1
                continue

        # 6b) Without dot, require uppercase start: "1 Introduction"
        m_num = re.match(r"^(?P<num>\d+(?:\.\d+)*)\s+(?P<title>[A-ZÀ-ÖØ-ÞĀ-Ž].+)$", s)
        if m_num:
            num = m_num.group("num")
            title = m_num.group("title").strip()
            if is_valid_section_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}",
                    "line_number": i,
                    "number": num,
                    "title": title,
                    "pattern": "text_numbered",
                })
                i += 1
                continue

        # 7) Appendix headings: "A Experiment", "A.1 Detailed Baselines", "B.2 Prompt"
        m_app = re.match(r"^(?P<num>[A-Z](?:\.\d+)*)\.\s+(?P<title>.+)$", s)
        if not m_app:
            m_app = re.match(r"^(?P<num>[A-Z](?:\.\d+)*)\s+(?P<title>[A-ZÀ-ÖØ-ÞĀ-Ž].+)$", s)
        if m_app:
            num = m_app.group("num")
            title = m_app.group("title").strip()
            if is_valid_appendix_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}",
                    "line_number": i,
                    "number": num,
                    "title": title,
                    "pattern": "text_numbered",
                })
                i += 1
                continue

        i += 1

    return sections


# ============================================================================
# 8. MERGE SECTIONS FROM BOTH METHODS
# ============================================================================

def extract_all_sections(doc, full_text):
    """
    Run both font-based and text-based section detection,
    merge results, deduplicate, and return unified list.
    """
    font_sections = _extract_sections_by_font(doc)
    text_sections = _extract_sections_by_text(full_text)

    # Convert font sections: find their line_number in normalized text
    norm_lines = _get_normalized_lines(full_text)
    for fs in font_sections:
        heading_clean = re.sub(r"\s+", " ", fs["heading"]).strip().lower()
        best_line = -1
        best_score = 0
        for li, ln in enumerate(norm_lines):
            ln_clean = re.sub(r"\s+", " ", ln).strip().lower()
            # Check if this line contains the heading text
            if heading_clean in ln_clean or ln_clean in heading_clean:
                score = len(heading_clean)
                if ln_clean == heading_clean:
                    score += 1000  # exact match bonus
                if score > best_score:
                    best_score = score
                    best_line = li
        fs["line_number"] = best_line if best_line >= 0 else 0

    # Merge: start with text sections, add font sections that are new
    all_secs = list(text_sections)
    
    # Build a set of existing section identifiers for dedup
    existing = set()
    for s in all_secs:
        existing.add(_norm_key(s))

    for fs in font_sections:
        key = _norm_key(fs)
        if key not in existing:
            all_secs.append(fs)
            existing.add(key)

    # Sort by line number
    all_secs.sort(key=lambda s: s.get("line_number", 0))

    # Final dedup pass and filtering
    return _postprocess_sections(all_secs)


def _norm_key(s):
    """Normalize a section for deduplication."""
    title = s.get("title", "")
    num = s.get("number", "") or ""
    # Normalize: lowercase, strip whitespace
    t = re.sub(r"\s+", " ", title.lower().strip())
    n = num.strip()
    # Key by number if available, else by title
    if n:
        return f"{n}|{t}"
    return f"|{t}"


def _postprocess_sections(sections):
    """Remove duplicates, references, checklist items."""
    seen = set()
    result = []
    
    # Detect if we're in the checklist section (NeurIPS papers)
    in_checklist = False
    
    for s in sections:
        title_lower = s["title"].lower().strip()
        
        # Skip references/bibliography
        if title_lower in _EXCLUDE_HEADINGS:
            continue
        
        # Detect NeurIPS checklist start
        if "neurips" in title_lower and "checklist" in title_lower:
            in_checklist = True
            continue
        if in_checklist:
            # Skip all checklist items (numbered 1-16 with generic titles)
            if s.get("number") and re.match(r"^\d+$", str(s["number"])):
                num_val = int(s["number"])
                if num_val >= 1 and num_val <= 20:
                    # Check if title looks like checklist (generic questions)
                    if any(kw in title_lower for kw in [
                        "claims", "limitations", "theory", "reproducibility",
                        "open access", "setting", "significance", "compute",
                        "ethics", "impacts", "safeguards", "licenses",
                        "assets", "crowdsourcing", "review board", "declaration",
                    ]):
                        continue
        
        # Dedup
        key = _norm_key(s)
        if key in seen:
            continue
        seen.add(key)
        
        # Also dedup by title alone (catches "Abstract" appearing in different patterns)
        title_key = re.sub(r"\s+", " ", title_lower)
        if title_key in seen:
            continue
        seen.add(title_key)
        
        result.append(s)
    
    return result


# ============================================================================
# 9. SECTION CONTENT EXTRACTION
# ============================================================================

def extract_sections_content(full_text, sections):
    lines = _get_normalized_lines(full_text)
    results = []
    for i, sec in enumerate(sections):
        start = sec.get("line_number", 0)
        end = sections[i + 1].get("line_number", len(lines)) if i + 1 < len(sections) else len(lines)
        content = "\n".join(lines[start + 1:end])
        results.append({**sec, "content": content.strip()})
    return results


# ============================================================================
# 10. STRUCTURED TEXT EXTRACTION
# ============================================================================

def extract_structured_text(pdf_path, structure):
    docs = []
    if structure["title"] or structure["authors"]:
        meta = "## Paper Metadata\n\n"
        if structure["title"]:
            meta += f"**Title:** {structure['title']}\n\n"
        if structure["authors"]:
            meta += f"**Authors:** {', '.join(structure['authors'])}\n\n"
        if structure["emails"]:
            meta += f"**Contact:** {', '.join(structure['emails'])}\n\n"
        if structure["organizations"]:
            meta += f"**Affiliations:** {'; '.join(structure['organizations'])}\n\n"
        docs.append({"type": "metadata", "content_type": "metadata", "text": meta})

    if structure["abstract"]:
        docs.append({
            "type": "abstract", "content_type": "section",
            "section_name": "Abstract",
            "text": f"## Abstract\n\n{structure['abstract']}",
        })

    doc = fitz.open(pdf_path)
    full_text = ""
    for p in range(len(doc)):
        full_text += doc[p].get_text("text") + "\n\n"
    doc.close()

    if structure["sections"]:
        secs = extract_sections_content(full_text, structure["sections"])
        for sec in secs:
            if sec["content"] and len(sec["content"]) > 50:
                name = sec.get("title", sec["heading"])
                if name.lower().strip() == "abstract" and structure.get("abstract"):
                    continue
                docs.append({
                    "type": "section", "content_type": "section",
                    "section_name": name,
                    "section_number": sec.get("number", ""),
                    "text": f"## {sec['heading']}\n\n{clean_text(sec['content'])}",
                })
    return docs


# ============================================================================
# 11. PAGE-BY-PAGE TEXT
# ============================================================================

def extract_text_pages(pdf_path):
    docs = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            docs.append({"type": "text", "page": i + 1, "text": clean_text(text)})
    doc.close()
    return docs


# ============================================================================
# 12. REFERENCES
# ============================================================================

def extract_references(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for p in range(len(doc)):
        full_text += doc[p].get_text("text") + "\n"
    doc.close()
    for pat in [r"(?:^|\n)\s*References?\s*\n", r"(?:^|\n)\s*REFERENCES?\s*\n",
                r"(?:^|\n)\s*Bibliography\s*\n"]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            ref_text = full_text[m.end():]
            for ep in [r"\n\s*(?:Appendix|APPENDIX|Supplementary|SUPPLEMENTARY)"]:
                me = re.search(ep, ref_text)
                if me:
                    ref_text = ref_text[:me.start()]
                    break
            refs = []
            matches = re.findall(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", ref_text, re.DOTALL)
            if matches:
                for num, txt in matches:
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if len(txt) > 20:
                        refs.append({"type": "reference", "number": num, "text": txt})
            else:
                current, num = "", 1
                for line in ref_text.split("\n"):
                    line = line.strip()
                    if not line:
                        if current:
                            refs.append({"type": "reference", "number": str(num), "text": current})
                            num += 1
                            current = ""
                    else:
                        current += (" " + line) if current else line
                if current:
                    refs.append({"type": "reference", "number": str(num), "text": current})
            return refs
    return []


# ============================================================================
# 13. TABLES
# ============================================================================

def _is_ref_table(text):
    if not text.strip():
        return False
    pats = [r"arXiv preprint", r"arXiv:\d+", r"In Proceedings",
            r"Journal of", r"et al\.", r"pages \d+[–\-]\d+"]
    total = hits = 0
    for l in text.split("\n"):
        l = l.strip()
        if not l or l == "nan":
            continue
        total += 1
        if any(re.search(p, l, re.IGNORECASE) for p in pats):
            hits += 1
    return total > 0 and hits / total > 0.3


def _is_junk_table(text):
    lines = [l.strip() for l in text.split("\n") if l.strip() and l.strip() != "nan"]
    if len(lines) < 3:
        return True
    at = " ".join(lines)
    alpha = sum(c.isalpha() for c in at)
    digit = sum(c.isdigit() or c in ".,−+-±·" for c in at)
    if digit > 0 and alpha < 30 and digit > alpha:
        return True
    cs = [r"\bDialogue\s+\d+", r"\bk\s*values\b", r"\(a\)\s+\w+", r"\(b\)\s+\w+", r"\bA-mem\b", r"\bBase\b"]
    if sum(1 for p in cs if re.search(p, at, re.IGNORECASE)) >= 2 and len(at) < 500:
        return True
    if re.search(r"\[Yes\]|\[No\]|\[NA\]|Justification:|Guidelines:", at):
        return True
    ls = [r"\bNote Construction\b", r"\bLink Generation\b", r"\bMemory Evolution\b",
          r"\bMemory Retrieval\b", r"\bInteraction\b", r"\bEnvironment\b", r"\bBox\s*[\dijn]"]
    if sum(1 for p in ls if re.search(p, at)) >= 2:
        return True
    pl = [l for l in lines if "|" in l]
    if pl:
        sc = sum(1 for l in pl if l.count("|") <= 3)
        if sc > len(pl) * 0.8 and len(at) < 600:
            return True
    if at.count("|") / max(len(at), 1) > 0.3 and len(at) < 200:
        return True
    return False


def extract_tables(pdf_path):
    tables = []
    if tabula is not None:
        for mode, label in [("lattice", "Tabula-Lattice"), ("stream", "Tabula-Stream")]:
            try:
                kw = {"lattice": True} if mode == "lattice" else {"stream": True}
                dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, silent=True, **kw)
                for idx, df in enumerate(dfs):
                    if df.empty:
                        continue
                    txt = df.to_markdown(index=False) if pd else df.to_string(index=False)
                    if txt.strip() and not _is_ref_table(txt) and not _is_junk_table(txt):
                        tables.append({"type": "table", "method": label, "index": idx + 1, "text": txt})
            except Exception as e:
                print(f"  ⚠ {label}: {e}")
    if pdfplumber is not None and pd is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for pn, page in enumerate(pdf.pages):
                    for idx, tbl in enumerate(page.extract_tables()):
                        if tbl:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0] if tbl[0] else None)
                            txt = df.to_markdown(index=False)
                            if txt.strip() and not _is_ref_table(txt) and not _is_junk_table(txt):
                                tables.append({"type": "table", "method": "PDFPlumber",
                                               "page": pn + 1, "index": idx + 1, "text": txt})
        except Exception as e:
            print(f"  ⚠ PDFPlumber: {e}")
    try:
        doc = fitz.open(pdf_path)
        for pn in range(len(doc)):
            for idx, tbl in enumerate(doc[pn].find_tables()):
                df = tbl.to_pandas()
                if df.empty:
                    continue
                txt = df.to_markdown(index=False) if pd else df.to_string(index=False)
                if txt.strip() and not _is_ref_table(txt) and not _is_junk_table(txt):
                    tables.append({"type": "table", "method": "PyMuPDF",
                                   "page": pn + 1, "index": idx + 1, "text": txt})
        doc.close()
    except Exception as e:
        print(f"  ⚠ PyMuPDF tables: {e}")
    return tables


# ============================================================================
# 14. FIGURES (page-level rendering)
# ============================================================================

def extract_figures_from_pdf(pdf_path):
    if Image is None:
        return []
    doc = fitz.open(pdf_path)
    figs = []
    for pn in range(len(doc)):
        page = doc[pn]
        pt = page.get_text("text")
        caps = re.findall(r"(Figure\s+\d+\s*[:.].*?)(?=\n|$)", pt, re.IGNORECASE)
        if caps:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            figs.append({
                "page": pn + 1, "width": pix.width, "height": pix.height,
                "image": img, "bytes": pix.tobytes("png"),
                "captions": [c.strip()[:200] for c in caps],
            })
    doc.close()
    return figs


# ============================================================================
# 15. OCR
# ============================================================================

def run_ocr(figs):
    if not figs or np is None:
        return []
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except ImportError:
        print("  ⚠ easyocr not installed")
        return []
    results = []
    for fig in figs:
        try:
            im = fig["image"]
            if im.mode != "RGB":
                im = im.convert("RGB")
            rs = reader.readtext(np.array(im))
            text = " ".join(r[1] for r in rs)
            if text.strip():
                results.append({"type": "image_ocr", "page": fig["page"], "text": clean_text(text)})
        except Exception:
            pass
    return results


# ============================================================================
# 16. IMAGE DESCRIPTIONS (Gemini Vision)
# ============================================================================

def create_image_descriptions(figs, api_key, title="", abstract=""):
    if not figs:
        print("  ⚠ No figures")
        return []
    if not api_key:
        print("  ⚠ No GOOGLE_API_KEY")
        return []
    print(f"  API key: {api_key[:10]}...")

    genai_mod = types_mod = sdk = None
    try:
        from google import genai as _g
        from google.genai import types as _t
        genai_mod, types_mod, sdk = _g, _t, "new"
    except ImportError:
        try:
            import google.generativeai as _old
            genai_mod, sdk = _old, "legacy"
        except ImportError:
            print("  ⚠ No genai SDK")
            return []

    def _prompt(pn, caps):
        c = ("\nCaptions:\n" + "\n".join(f"- {x}" for x in caps)) if caps else ""
        return f"""Analyze this page from "{title}".{c}

Describe ONLY figures/diagrams (ignore body text).
For each figure: number, type, sub-figures, components, labels, values, meaning.
If no figures, respond: SKIP"""

    results = []
    if sdk == "new":
        try:
            cl = genai_mod.Client(api_key=api_key)
        except Exception as e:
            print(f"  ⚠ {e}")
            return []
        for idx, fig in enumerate(figs, 1):
            try:
                print(f"    Vision {idx}/{len(figs)} (p{fig['page']})...", end=" ")
                r = cl.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types_mod.Content(role="user", parts=[
                        types_mod.Part.from_bytes(data=fig["bytes"], mime_type="image/png"),
                        types_mod.Part.from_text(text=_prompt(fig["page"], fig.get("captions", []))),
                    ])])
                d = r.text
                if d and not d.strip().upper().startswith("SKIP"):
                    results.append({"type": "image_description", "page": fig["page"],
                                    "text": f"Figure from page {fig['page']}: {d}"})
                    print("✅")
                else:
                    print("⏭")
            except Exception as e:
                print(f"❌ {e}")
    elif sdk == "legacy":
        genai_mod.configure(api_key=api_key)
        mdl = genai_mod.GenerativeModel("gemini-2.0-flash")
        for idx, fig in enumerate(figs, 1):
            try:
                print(f"    Vision {idx}/{len(figs)} (p{fig['page']})...", end=" ")
                im = fig["image"]
                if im.mode != "RGB":
                    im = im.convert("RGB")
                r = mdl.generate_content([_prompt(fig["page"], fig.get("captions", [])), im])
                d = r.text
                if d and not d.strip().upper().startswith("SKIP"):
                    results.append({"type": "image_description", "page": fig["page"],
                                    "text": f"Figure from page {fig['page']}: {d}"})
                    print("✅")
                else:
                    print("⏭")
            except Exception as e:
                print(f"❌ {e}")
    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_document_structure(pdf_path):
    doc = fitz.open(pdf_path)
    header_text = ""
    for i in range(min(3, len(doc))):
        header_text += doc[i].get_text("text") + "\n"
    full_text = ""
    for i in range(len(doc)):
        full_text += doc[i].get_text("text") + "\n"

    title = extract_title(doc, header_text)
    
    # Dual section detection
    sections = extract_all_sections(doc, full_text)
    
    structure = {
        "title": title,
        "authors": extract_authors(header_text, title),
        "emails": extract_emails(header_text),
        "organizations": extract_organizations(header_text),
        "abstract": extract_abstract(header_text),
        "sections": sections,
        "num_pages": len(doc),
    }
    doc.close()
    return structure


def run_full_analysis(cfg):
    rpt = ReportWriter()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rpt.header(f"DOCUMENT PARSING REPORT — {ts}")
    rpt.kv("PDF", cfg.PDF_PATH)
    rpt.kv("Skip Vision", cfg.SKIP_VISION)
    rpt.kv("Skip OCR", cfg.SKIP_OCR)
    rpt.line()

    rpt.header("1. STRUCTURE ANALYSIS")
    structure = analyze_document_structure(cfg.PDF_PATH)

    rpt.kv("Pages", structure["num_pages"])
    rpt.kv("Title", structure["title"] or "(not detected)")
    rpt.kv("Authors", len(structure["authors"]))
    for a in structure["authors"]:
        rpt.line(f"      • {a}")
    rpt.kv("Emails", len(structure["emails"]))
    for e in structure["emails"]:
        rpt.line(f"      • {e}")
    rpt.kv("Organizations", len(structure["organizations"]))
    for o in structure["organizations"]:
        rpt.line(f"      • {o}")
    rpt.kv("Abstract length", len(structure["abstract"]))
    if structure["abstract"]:
        rpt.subheader("Abstract preview")
        rpt.preview(structure["abstract"], 600)

    rpt.header("2. SECTION HEADINGS")
    rpt.kv("Total sections", len(structure["sections"]))
    rpt.line()
    by_pat = {}
    for s in structure["sections"]:
        by_pat.setdefault(s["pattern"], []).append(s)
    for pat, secs in by_pat.items():
        rpt.subheader(f"Pattern: {pat}  ({len(secs)} found)")
        for s in secs:
            num = s.get("number") or "---"
            rpt.line(f"    Line {s.get('line_number', '?'):>5}  [{num:>8}]  {s['title']}")

    rpt.header("3. STRUCTURED DOCUMENTS")
    struct_docs = extract_structured_text(cfg.PDF_PATH, structure)
    rpt.kv("Documents created", len(struct_docs))
    rpt.line()
    for i, d in enumerate(struct_docs):
        label = d.get("section_name", d["type"])
        rpt.subheader(f"Doc {i+1}: [{d['content_type']}] {label}")
        rpt.kv("Type", d["type"])
        rpt.kv("Chars", len(d["text"]))
        rpt.preview(d["text"], 400)
        rpt.line()

    rpt.header("4. PAGE-BY-PAGE TEXT")
    page_docs = extract_text_pages(cfg.PDF_PATH)
    rpt.kv("Pages with text", len(page_docs))
    for d in page_docs:
        rpt.line(f"    Page {d['page']:2d}: {len(d['text']):5d} chars")
    for d in page_docs[:2]:
        rpt.subheader(f"Page {d['page']} preview")
        rpt.preview(d["text"], 300)

    rpt.header("5. REFERENCES")
    refs = extract_references(cfg.PDF_PATH)
    rpt.kv("References found", len(refs))
    for r in refs[:10]:
        rpt.line(f"    [{r['number']:>3}] {r['text'][:120]}...")
    if len(refs) > 10:
        rpt.line(f"    ... and {len(refs) - 10} more")

    rpt.header("6. TABLES")
    tables = extract_tables(cfg.PDF_PATH)
    rpt.kv("Tables (after junk filter)", len(tables))
    by_m = {}
    for t in tables:
        by_m.setdefault(t["method"], []).append(t)
    for method, tbls in by_m.items():
        rpt.subheader(f"{method} ({len(tbls)} tables)")
        for t in tbls:
            rpt.line(f"    Table {t['index']}: {len(t['text'])} chars")
            rpt.preview(t["text"], 300)
            rpt.line()

    rpt.header("7. FIGURES")
    figs = extract_figures_from_pdf(cfg.PDF_PATH)
    rpt.kv("Figure pages", len(figs))
    for f in figs:
        c = "; ".join(f.get("captions", []))[:150]
        rpt.line(f"    Page {f['page']}: {f['width']}x{f['height']}  | {c}")

    rpt.header("8. OCR")
    if cfg.SKIP_OCR:
        rpt.line("  (skipped)")
        ocr_docs = []
    else:
        ocr_docs = run_ocr(figs)
        rpt.kv("OCR docs", len(ocr_docs))
        for d in ocr_docs:
            rpt.subheader(f"OCR — page {d['page']}")
            rpt.preview(d["text"], 300)

    rpt.header("9. VISION")
    if cfg.SKIP_VISION:
        rpt.line("  (skipped)")
        vision_docs = []
    else:
        vision_docs = create_image_descriptions(
            figs, cfg.GOOGLE_API_KEY,
            title=structure.get("title", ""), abstract=structure.get("abstract", ""))
        rpt.kv("Vision descriptions", len(vision_docs))
        for d in vision_docs:
            rpt.subheader(f"Vision — page {d['page']}")
            rpt.preview(d["text"], 800)
            rpt.line()

    rpt.header("10. DIAGNOSTICS")
    ts2 = len(struct_docs)
    tp = len(page_docs)
    rpt.kv("Structured docs", ts2)
    rpt.kv("Page-by-page docs", tp)
    rpt.line()
    rpt.line("  ⚠ Recommend ONLY structured docs (no page-by-page) to avoid duplication.")
    rpt.line()

    issues = False
    if not structure["title"] or _TITLE_REJECT.search(structure["title"]):
        rpt.line("  ⚠ Title may be incorrect")
        issues = True
    if not structure["authors"]:
        rpt.line("  ⚠ No authors detected")
        issues = True
    stl = {s["title"].lower() for s in structure["sections"]}
    for exp in ["introduction", "conclusion", "conclusions"]:
        if not any(exp in t for t in stl):
            rpt.line(f"  ⚠ Missing expected section: '{exp}'")
            issues = True
    if not issues:
        rpt.line("  ✅ No major issues detected")

    rpt.subheader("Chunking plan")
    rpt.line(f"    Sections:   {cfg.SEC_CHUNK_SIZE}/{cfg.SEC_CHUNK_OVERLAP}")
    rpt.line(f"    Tables:     {cfg.TABLE_CHUNK_SIZE}/{cfg.TABLE_CHUNK_OVERLAP}")
    rpt.line(f"    Figures:    {cfg.FIGURE_CHUNK_SIZE}/{cfg.FIGURE_CHUNK_OVERLAP}")
    rpt.line(f"    References: {cfg.REF_CHUNK_SIZE}/{cfg.REF_CHUNK_OVERLAP}")
    rpt.line()
    total = ts2 + len(tables) + len(ocr_docs) + len(vision_docs) + len(refs)
    rpt.kv("Total docs (recommended)", total)
    rpt.kv("Total docs (with page duplication)", total + tp)

    rpt.save(cfg.OUTPUT_PATH)


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("--output", "-o", default="parsing_report.txt")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.pdf_path):
        sys.exit(f"Not found: {args.pdf_path}")

    cfg = Config(
        PDF_PATH=args.pdf_path, OUTPUT_PATH=args.output,
        SKIP_VISION=args.skip_vision, SKIP_OCR=args.skip_ocr,
        GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY", ""),
    )
    run_full_analysis(cfg)


if __name__ == "__main__":
    main()