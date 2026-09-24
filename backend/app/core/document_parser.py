"""
Document parsing module — Dual section detection (font + text-regex), merged and deduplicated.

Contains:
- Structure analysis (title, authors, emails, organizations, abstract, section headings)
- Dual section detection: font analysis + text regex, merged and deduplicated
- Structured text extraction (metadata, abstract, sections as separate docs)
- Full text extraction (page-by-page)
- Reference extraction and parsing
- Table extraction (Docling layout + TableFormer, with junk filter)
- Figure extraction (page-level rendering with captions)
- OCR (per-figure-page)
- Image description generation (Gemini Vision with context-aware prompt)
- Complete document collection pipeline
"""

import io
import re
import logging
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import numpy as np
from docling.document_converter import DocumentConverter
from PIL import Image
from google import genai
from google.genai import types
from llama_index.core.schema import Document

from app.config import config
from app.utils.text_cleaning import clean_text, clean_text_inline

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED PATTERNS
# =============================================================================

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


# =============================================================================
# 1. TITLE EXTRACTION
# =============================================================================

def extract_title(doc, header_text: str) -> str:
    """Extract paper title using font analysis with reject filter and multi-span merging."""
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
        # Merge consecutive spans at the same font size (split titles)
        for sz, txt in candidates[1:]:
            if abs(sz - top) < 1.0 and len(title) < 150:
                title += " " + txt
            else:
                break
        return clean_text_inline(title)

    # Fallback: Use first substantial line
    for ln in header_text.split("\n")[:15]:
        ln = ln.strip()
        if len(ln) > 20 and not _TITLE_REJECT.search(ln):
            if not re.match(r"^(Abstract|Introduction|\d+\.)", ln, re.IGNORECASE):
                return clean_text_inline(ln)

    return ""


# =============================================================================
# 2. AUTHORS
# =============================================================================

def extract_authors(header_text: str, title: str = "") -> List[str]:
    """Extract author names — Unicode-aware with org/title rejection."""
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


# =============================================================================
# 3. EMAILS
# =============================================================================

def extract_emails(header_text: str) -> List[str]:
    """Extract email addresses from text."""
    return list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", header_text)))


# =============================================================================
# 4. ORGANIZATIONS
# =============================================================================

def extract_organizations(header_text: str) -> List[str]:
    """Extract organizations with email/sentence guards and expanded keywords."""
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


# =============================================================================
# 5. ABSTRACT
# =============================================================================

def extract_abstract(header_text: str) -> str:
    """Extract the abstract section."""
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


# =============================================================================
# 6. SECTION DETECTION — METHOD A: FONT ANALYSIS
# =============================================================================

def _extract_sections_by_font(doc) -> List[Dict[str, Any]]:
    """
    Extract section headings by analyzing font properties (size, bold).
    Strategy: find body text size, then identify spans that are larger or bolder.
    """
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

    # Determine body text size (most frequent font size by total character count)
    size_counts: Counter = Counter()
    for s in all_spans:
        if len(s["text"]) > 20:
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
        if s["size"] > body_size + 0.5:
            is_heading_font = True
        if s["bold"] and abs(s["size"] - body_size) < 1.5:
            is_heading_font = True

        if not is_heading_font:
            continue
        if len(text) > 150:
            continue
        if s["size"] > body_size + 4 and s["page"] == 0:
            continue
        if "@" in text or "http" in text.lower() or _TITLE_REJECT.search(text):
            continue

        heading_candidates.append({
            "text": text,
            "size": s["size"],
            "bold": s["bold"],
            "page": s["page"],
            "y": s["y"],
        })

    # Parse heading candidates into structured sections
    sections = []
    for h in heading_candidates:
        text = h["text"].strip()

        # Try numbered heading: "1 Introduction", "3.2.1 Scaled..."
        m = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", text)
        if not m:
            m = re.match(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", text)
        if not m:
            # Appendix: "A Experiment", "B.1 Prompt Template"
            m = re.match(r"^([A-Z](?:\.\d+)*)\s+(.+)$", text)

        if m:
            num = m.group(1)
            title = m.group(2).strip()

            if re.match(r"^\d+(?:\.\d+)*$", num):
                parts = num.split(".")
                if len(parts[0]) > 2 or len(parts) > 4:
                    continue
                if any(len(p) > 2 for p in parts[1:]):
                    continue
                skip = False
                for p in parts[1:]:
                    if int(p) > 20:
                        skip = True
                        break
                if skip:
                    continue
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
                "heading": text, "number": num, "title": title,
                "page": h["page"], "y": h["y"],
                "pattern": "font_numbered", "size": h["size"], "bold": h["bold"],
            })
        else:
            # Unnumbered heading
            if len(text) > 80 or not text[0].isupper():
                continue
            if re.match(r"^(Table|Figure)\s+\d+", text, re.IGNORECASE):
                continue
            if text.lower() in ("preprint", "preprint."):
                continue

            sections.append({
                "heading": text, "number": None, "title": text,
                "page": h["page"], "y": h["y"],
                "pattern": "font_unnumbered", "size": h["size"], "bold": h["bold"],
            })

    return sections


# =============================================================================
# 7. SECTION DETECTION — METHOD B: TEXT REGEX
# =============================================================================

def _get_normalized_lines(full_text: str) -> List[str]:
    """Normalize text into clean single-space lines, dropping empty ones."""
    lines = []
    for ln in full_text.splitlines():
        s = re.sub(r"\s+", " ", (ln or "")).strip()
        if s:
            lines.append(s)
    return lines


def _extract_sections_by_text(full_text: str) -> List[Dict[str, Any]]:
    """
    Section extraction using text patterns — numbered headings, ALL_CAPS,
    keyword headings, table/figure captions, appendix headings.
    """
    acronym_exclusions = {
        'BLEU', 'GPU', 'CPU', 'LSTM', 'RNN', 'CNN', 'NLP', 'WMT', 'TPU',
        'ADAM', 'SGD', 'RELU', 'GELU', 'FFN', 'MLP', 'BERT', 'GPT', 'LLM',
        'PPL', 'GNMT', 'API', 'URL', 'HTTP', 'HTML', 'JSON', 'XML', 'SQL',
        'ROUGE', 'METEOR', 'SBERT', 'RAM', 'ROM',
        'LOCOMO', 'READAGENT', 'MEMORYBANK', 'MEMGPT', 'AMEM',
        'LLAMA', 'QWEN', 'DEEPSEEK',
    }

    def is_number_only(s):
        return re.match(r"^\d+(?:\.\d+)*\.?$", s) is not None

    def is_appendix_number_only(s):
        return re.match(r"^[A-Z](?:\.\d+)*\.?$", s) is not None

    def is_valid_section_number(num):
        if not re.match(r"^\d+(?:\.\d+)*$", num):
            return False
        parts = num.split(".")
        if len(parts) > 4 or len(parts[0]) > 2:
            return False
        if any(len(p) > 2 for p in parts[1:]):
            return False
        for p in parts[1:]:
            if int(p) > 20:
                return False
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
        if re.search(r"https?://|arxiv|@", nxt, re.IGNORECASE):
            return False
        alpha_count = sum(c.isalpha() for c in nxt)
        if alpha_count > 30:
            return False
        digit_ratio = sum(c.isdigit() or c in ".,·±−+% " for c in nxt) / max(len(nxt), 1)
        return digit_ratio > 0.5

    lines = _get_normalized_lines(full_text)
    sections = []
    in_references = False

    i = 0
    while i < len(lines):
        s = lines[i]

        if in_references:
            i += 1
            continue

        # 1) Table/Figure captions
        if is_table_caption(s):
            m = re.match(r"^(?P<pre>Table|TABLE|Figure|FIGURE)\s+(?P<num>\d+)\s*[:.]\s*(?P<title>.+)$", s)
            if m:
                pre = m.group("pre").capitalize()
                sections.append({
                    "heading": s, "line_number": i,
                    "number": f"{pre} {m.group('num')}",
                    "title": f"{pre} {m.group('num')}: {m.group('title').strip()}",
                    "pattern": "text_caption",
                })
            i += 1
            continue

        # 2) Merge split headers: number on one line, title on next
        if is_number_only(s) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if looks_like_title(nxt):
                s = f"{s.rstrip('.')} {nxt}"
                i += 1

        # 2b) Same for appendix numbers
        if is_appendix_number_only(s) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if looks_like_title(nxt):
                s = f"{s.rstrip('.')} {nxt}"
                i += 1

        # 3) Skip reference entries
        if is_reference_entry(s):
            i += 1
            continue

        # 4) Keyword headings
        if s.strip().lower() in _KEYWORD_HEADINGS:
            sections.append({
                "heading": s.strip(), "line_number": i,
                "number": None, "title": s.strip(),
                "pattern": "text_keyword",
            })
            i += 1
            continue

        # 4b) References boundary — set flag, don't add as section
        if s.strip().lower() in ("references", "bibliography", "reference"):
            in_references = True
            i += 1
            continue

        # 5) ALL CAPS headings
        m_caps = re.match(r"^([A-Z]{3,}(?:\s+[A-Z]{3,})*)$", s)
        if m_caps:
            heading = m_caps.group(1)
            if heading not in acronym_exclusions and len(heading) >= 5:
                if not next_line_is_data(i):
                    sections.append({
                        "heading": heading, "line_number": i,
                        "number": None, "title": heading,
                        "pattern": "text_allcaps",
                    })
                    if heading.lower() == "references":
                        in_references = True
            i += 1
            continue

        # 6) Numbered headings with dot: "1. Introduction"
        m_num = re.match(r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+)$", s)
        if m_num:
            num = m_num.group("num")
            title = m_num.group("title").strip()
            if is_valid_section_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}", "line_number": i,
                    "number": num, "title": title,
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
                    "heading": f"{num} {title}", "line_number": i,
                    "number": num, "title": title,
                    "pattern": "text_numbered",
                })
                i += 1
                continue

        # 7) Appendix headings
        m_app = re.match(r"^(?P<num>[A-Z](?:\.\d+)*)\.\s+(?P<title>.+)$", s)
        if not m_app:
            m_app = re.match(r"^(?P<num>[A-Z](?:\.\d+)*)\s+(?P<title>[A-ZÀ-ÖØ-ÞĀ-Ž].+)$", s)
        if m_app:
            num = m_app.group("num")
            title = m_app.group("title").strip()
            if is_valid_appendix_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}", "line_number": i,
                    "number": num, "title": title,
                    "pattern": "text_numbered",
                })
                i += 1
                continue

        i += 1

    return sections


# =============================================================================
# 8. MERGE SECTIONS FROM BOTH METHODS
# =============================================================================

def _norm_key(s: Dict) -> str:
    """Normalize a section for deduplication."""
    title = s.get("title", "")
    num = s.get("number", "") or ""
    t = re.sub(r"\s+", " ", title.lower().strip())
    n = num.strip()
    return f"{n}|{t}" if n else f"|{t}"


def _postprocess_sections(sections: List[Dict]) -> List[Dict]:
    """Remove duplicates, references, checklist items."""
    seen = set()
    result = []
    in_checklist = False

    for s in sections:
        title_lower = s["title"].lower().strip()

        if title_lower in _EXCLUDE_HEADINGS:
            continue

        # NeurIPS checklist detection
        if "neurips" in title_lower and "checklist" in title_lower:
            in_checklist = True
            continue
        if in_checklist:
            if s.get("number") and re.match(r"^\d+$", str(s["number"])):
                num_val = int(s["number"])
                if 1 <= num_val <= 20:
                    if any(kw in title_lower for kw in [
                        "claims", "limitations", "theory", "reproducibility",
                        "open access", "setting", "significance", "compute",
                        "ethics", "impacts", "safeguards", "licenses",
                        "assets", "crowdsourcing", "review board", "declaration",
                    ]):
                        continue

        key = _norm_key(s)
        if key in seen:
            continue
        seen.add(key)

        title_key = re.sub(r"\s+", " ", title_lower)
        if title_key in seen:
            continue
        seen.add(title_key)

        result.append(s)

    return result


def extract_all_sections(doc, full_text: str) -> List[Dict[str, Any]]:
    """
    Run both font-based and text-based section detection,
    merge results, deduplicate, and return unified list.
    """
    font_sections = _extract_sections_by_font(doc)
    text_sections = _extract_sections_by_text(full_text)

    # Map font sections to line numbers in the normalized text
    norm_lines = _get_normalized_lines(full_text)
    for fs in font_sections:
        heading_clean = re.sub(r"\s+", " ", fs["heading"]).strip().lower()
        best_line = -1
        best_score = 0
        for li, ln in enumerate(norm_lines):
            ln_clean = re.sub(r"\s+", " ", ln).strip().lower()
            if heading_clean in ln_clean or ln_clean in heading_clean:
                score = len(heading_clean)
                if ln_clean == heading_clean:
                    score += 1000
                if score > best_score:
                    best_score = score
                    best_line = li
        fs["line_number"] = best_line if best_line >= 0 else 0

    # Merge: text sections first, add new font sections
    all_secs = list(text_sections)
    existing = set(_norm_key(s) for s in all_secs)

    for fs in font_sections:
        key = _norm_key(fs)
        if key not in existing:
            all_secs.append(fs)
            existing.add(key)

    all_secs.sort(key=lambda s: s.get("line_number", 0))
    return _postprocess_sections(all_secs)


# =============================================================================
# 9. SECTION CONTENT EXTRACTION
# =============================================================================

def extract_sections_content(full_text: str, sections: List[Dict]) -> List[Dict]:
    """Extract content for each section using normalized lines."""
    lines = _get_normalized_lines(full_text)
    results = []
    for i, sec in enumerate(sections):
        start = sec.get("line_number", 0)
        end = sections[i + 1].get("line_number", len(lines)) if i + 1 < len(sections) else len(lines)
        content = "\n".join(lines[start + 1:end])
        results.append({**sec, "content": content.strip()})
    return results


# =============================================================================
# 10. STRUCTURE ANALYSIS
# =============================================================================

def analyze_document_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF structure to extract title, authors, emails, organizations,
    abstract, and section headings using dual detection.
    """
    logger.info("📋 Analyzing document structure...")

    doc = fitz.open(pdf_path)

    header_text = ""
    for page_num in range(min(3, len(doc))):
        header_text += doc[page_num].get_text("text") + "\n"

    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc[page_num].get_text("text") + "\n"

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

    logger.info(f"   📌 Title: {structure['title'][:60]}..." if len(structure['title']) > 60 else f"   📌 Title: {structure['title']}")
    logger.info(f"   👥 Authors: {len(structure['authors'])} found")
    logger.info(f"   📧 Emails: {len(structure['emails'])} found")
    logger.info(f"   🏢 Organizations: {len(structure['organizations'])} found")
    logger.info(f"   📝 Abstract: {'Found' if structure['abstract'] else 'Not found'}")
    logger.info(f"   📑 Sections: {len(structure['sections'])} found")

    return structure


# =============================================================================
# 11. STRUCTURED TEXT EXTRACTION
# =============================================================================

def extract_structured_text_from_pdf(pdf_path: str, structure: Dict[str, Any]) -> List[Document]:
    """
    Extract text with structure-aware metadata.
    Creates separate Documents for metadata, abstract, and each section.
    """
    logger.info("📄 Extracting structured text...")
    documents = []

    # Create metadata document
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

        documents.append(Document(
            text=meta,
            metadata={
                "source": pdf_path,
                "type": "metadata",
                "content_type": "metadata",
                "title": structure["title"],
                "authors": structure["authors"][:5] if structure["authors"] else [],
            }
        ))

    # Create abstract document
    if structure["abstract"]:
        documents.append(Document(
            text=f"## Abstract\n\n{structure['abstract']}",
            metadata={
                "source": pdf_path,
                "type": "abstract",
                "content_type": "section",
                "section_name": "Abstract",
            }
        ))

    # Extract text by sections
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc[page_num].get_text("text") + "\n\n"
    doc.close()

    if structure["sections"]:
        secs = extract_sections_content(full_text, structure["sections"])
        for sec in secs:
            if sec["content"] and len(sec["content"]) > 50:
                name = sec.get("title", sec["heading"])
                # Skip duplicate abstract
                if name.lower().strip() == "abstract" and structure.get("abstract"):
                    continue
                documents.append(Document(
                    text=clean_text(f"## {sec['heading']}\n\n{clean_text_inline(sec['content'])}"),
                    metadata={
                        "source": pdf_path,
                        "type": "section",
                        "content_type": "section",
                        "section_name": name,
                        "section_number": sec.get("number", ""),
                    }
                ))

    logger.info(f"   ✅ {len(documents)} structured documents created")
    return documents


# =============================================================================
# 12. FULL TEXT EXTRACTION (PAGE-BY-PAGE)
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """Extract text from PDF page by page."""
    logger.info("📄 Extracting full text (page by page)...")
    documents = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            cleaned = clean_text(text)
            if cleaned:
                documents.append(Document(
                    text=cleaned,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "text",
                        "content_type": "text",
                    }
                ))

    doc.close()
    logger.info(f"   ✅ {len(documents)} pages extracted")
    return documents


# =============================================================================
# 13. REFERENCE EXTRACTION
# =============================================================================

def extract_references_from_pdf(pdf_path: str) -> List[Document]:
    """Extract references/bibliography section and chunk into Documents."""
    logger.info("📚 Extracting references...")

    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc[page_num].get_text("text") + "\n"
    doc.close()

    # Find references section
    for pat in [
        r"(?:^|\n)\s*References?\s*\n",
        r"(?:^|\n)\s*REFERENCES?\s*\n",
        r"(?:^|\n)\s*Bibliography\s*\n",
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            ref_text = full_text[m.end():]
            # Find end of references
            for ep in [r"\n\s*(?:Appendix|APPENDIX|Supplementary|SUPPLEMENTARY)"]:
                me = re.search(ep, ref_text)
                if me:
                    ref_text = ref_text[:me.start()]
                    break

            # Parse individual references
            refs = []
            matches = re.findall(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", ref_text, re.DOTALL)
            if matches:
                for num, txt in matches:
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if len(txt) > 20:
                        refs.append({"number": num, "text": txt})
            else:
                current, num = "", 1
                for line in ref_text.split("\n"):
                    line = line.strip()
                    if not line:
                        if current:
                            refs.append({"number": str(num), "text": current})
                            num += 1
                            current = ""
                    else:
                        current += (" " + line) if current else line
                if current:
                    refs.append({"number": str(num), "text": current})

            if not refs:
                logger.info("   ⚠️ No individual references parsed")
                return []

            # Chunk references into documents
            documents = []
            ref_chunks = []
            current_chunk = []
            current_length = 0

            for ref in refs:
                ref_line = f"[{ref['number']}] {ref['text']}"
                ref_length = len(ref_line)
                if current_length + ref_length > config.REF_CHUNK_SIZE and current_chunk:
                    ref_chunks.append(current_chunk)
                    current_chunk = [ref_line]
                    current_length = ref_length
                else:
                    current_chunk.append(ref_line)
                    current_length += ref_length

            if current_chunk:
                ref_chunks.append(current_chunk)

            for i, chunk in enumerate(ref_chunks):
                chunk_text = "## References\n\n" + "\n\n".join(chunk)
                documents.append(Document(
                    text=chunk_text,
                    metadata={
                        "source": pdf_path,
                        "type": "references",
                        "content_type": "references",
                        "chunk_index": i + 1,
                        "total_chunks": len(ref_chunks),
                        "reference_count": len(chunk),
                    }
                ))

            logger.info(f"   ✅ {len(refs)} references extracted into {len(documents)} documents")
            return documents

    logger.info("   ⚠️ No references section found")
    return []


# =============================================================================
# 14. TABLE EXTRACTION (WITH JUNK FILTER)
# =============================================================================

def _is_ref_table(text: str) -> bool:
    """Check if a table is actually a references section."""
    if not text.strip():
        return False
    pats = [r"arXiv preprint", r"arXiv:\d+", r"In Proceedings",
            r"Journal of", r"et al\.", r"pages \d+[–\-]\d+"]
    total = hits = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line or line == "nan":
            continue
        total += 1
        if any(re.search(p, line, re.IGNORECASE) for p in pats):
            hits += 1
    return total > 0 and hits / total > 0.3


def _is_junk_table(text: str) -> bool:
    """Filter out junk tables: checklists, sparse pipes, digit-heavy noise."""
    lines = [l.strip() for l in text.split("\n") if l.strip() and l.strip() != "nan"]
    if len(lines) < 3:
        return True

    at = " ".join(lines)
    alpha = sum(c.isalpha() for c in at)
    digit = sum(c.isdigit() or c in ".,−+-±·" for c in at)

    if digit > 0 and alpha < 30 and digit > alpha:
        return True

    # NeurIPS checklist patterns
    if re.search(r"\[Yes\]|\[No\]|\[NA\]|Justification:|Guidelines:", at):
        return True

    # Sparse pipe tables
    pl = [l for l in lines if "|" in l]
    if pl:
        sc = sum(1 for l in pl if l.count("|") <= 3)
        if sc > len(pl) * 0.8 and len(at) < 600:
            return True
    if at.count("|") / max(len(at), 1) > 0.3 and len(at) < 200:
        return True

    return False


@lru_cache(maxsize=1)
def _docling_converter() -> DocumentConverter:
    return DocumentConverter()


def extract_tables_from_pdf(pdf_path: str) -> List[Document]:
    """Extract tables using Docling (layout analysis + TableFormer) with junk filtering."""
    logger.info("📊 Extracting tables (Docling)...")

    doc = _docling_converter().convert(pdf_path).document
    table_docs = []
    references_filtered = 0

    for table in doc.tables:
        table_text = table.export_to_markdown(doc=doc)
        if _is_ref_table(table_text):
            references_filtered += 1
            continue
        if _is_junk_table(table_text):
            continue

        table_docs.append(Document(
            text=table_text,
            metadata={
                "type": "table",
                "content_type": "table",
                "table_index": len(table_docs) + 1,
                "page": table.prov[0].page_no if table.prov else 0,
            }
        ))

    logger.info(f"   📋 References filtered: {references_filtered}")
    logger.info(f"   ✅ Total: {len(table_docs)} table documents")
    return table_docs


# =============================================================================
# 15. FIGURE EXTRACTION (PAGE-LEVEL RENDERING)
# =============================================================================

def extract_figures_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract figures by rendering pages that contain figure captions.
    Returns page-level images with caption metadata.
    """
    logger.info("🖼️ Extracting figures (page-level)...")
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
                "page": pn + 1,
                "width": pix.width,
                "height": pix.height,
                "image": img,
                "bytes": pix.tobytes("png"),
                "captions": [c.strip()[:200] for c in caps],
            })

    doc.close()
    logger.info(f"   ✅ {len(figs)} figure pages extracted")
    return figs


# =============================================================================
# 16. OCR    (PER-FIGURE-PAGE)
# =============================================================================

def ocr_images(figures: List[Dict]) -> List[Document]:
    """Run OCR on figure pages using EasyOCR."""
    if not figures:
        return []

    logger.info("🔍 Running OCR...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        logger.warning("   ⚠️ easyocr not installed, skipping OCR")
        return []

    ocr_docs = []
    for fig in figures:
        try:
            image = fig["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            results = reader.readtext(np.array(image))
            text = " ".join([r[1] for r in results])
            if text.strip():
                ocr_docs.append(Document(
                    text=f"Figure (page {fig['page']}): {clean_text_inline(text)}",
                    metadata={
                        "type": "image_ocr",
                        "content_type": "figure",
                        "page": fig["page"],
                    }
                ))
        except Exception:
            pass

    logger.info(f"   ✅ OCR completed for {len(ocr_docs)} figure pages")
    return ocr_docs


# =============================================================================
# 17. IMAGE DESCRIPTIONS (GEMINI VISION)
# =============================================================================

def create_image_descriptions(
    figures: List[Dict],
    title: str = "",
    abstract: str = "",
) -> List[Document]:
    """Create text descriptions for figure pages using Gemini Vision."""
    if not figures:
        return []

    logger.info("🎨 Creating image descriptions with Vision AI...")

    try:
        vision_client = genai.Client(api_key=config.GOOGLE_API_KEY)

        docs = []
        for idx, fig in enumerate(figures, 1):
            try:
                logger.info(f"   Processing {idx}/{len(figures)} (page {fig['page']})...")

                captions = fig.get("captions", [])
                cap_text = ("\nCaptions:\n" + "\n".join(f"- {x}" for x in captions)) if captions else ""

                prompt = f"""Analyze this page from "{title}".{cap_text}

Describe ONLY figures/diagrams (ignore body text).
For each figure: number, type, sub-figures, components, labels, values, meaning.
If no figures, respond: SKIP"""

                response = vision_client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(data=fig["bytes"], mime_type="image/png"),
                                types.Part.from_text(text=prompt),
                            ],
                        ),
                    ],
                )

                description = response.text

                if description and not description.strip().upper().startswith("SKIP"):
                    docs.append(Document(
                        text=f"Figure from page {fig['page']}: {description}",
                        metadata={
                            "type": "image_description",
                            "content_type": "figure",
                            "page": fig["page"],
                        }
                    ))
                    logger.info(f"   ✅ Page {fig['page']} described")
                else:
                    logger.info(f"   ⏭️ Page {fig['page']} — no figures detected")

            except Exception as e:
                logger.error(f"   ❌ Error on page {fig['page']}: {str(e)[:80]}")

        logger.info(f"   ✅ {len(docs)} descriptions created")
        return docs

    except Exception as e:
        logger.error(f"   ❌ Vision API error: {e}")
        return []


# =============================================================================
# 18. COMPLETE DOCUMENT COLLECTION PIPELINE
# =============================================================================

def collect_all_documents(pdf_path: str) -> List[Document]:
    """
    Complete document collection pipeline.

    Flow:
    1. Structure Analysis (title, authors, emails, organizations, abstract, sections)
    2. Structured Text Extraction (metadata, abstract, sections as separate docs)
    3. Full Text Extraction (page-by-page)
    4. Table Extraction (hybrid with junk filter)
    5. Figure Extraction (page-level rendering)
    6. OCR (per-figure-page)
    7. Image Descriptions (Gemini Vision with context)
    8. Reference Extraction
    """
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENT COLLECTION PIPELINE (ENHANCED)")
    logger.info("=" * 60)

    all_docs = []

    # Step 1: Structure Analysis
    structure = analyze_document_structure(pdf_path)

    # Step 2: Structured Text Extraction
    structured_docs = extract_structured_text_from_pdf(pdf_path, structure)
    all_docs.extend(structured_docs)

    # Step 3: Full Text Extraction
    text_docs = extract_text_from_pdf(pdf_path)
    all_docs.extend(text_docs)

    # Step 4: Table Extraction
    table_docs = extract_tables_from_pdf(pdf_path)
    all_docs.extend(table_docs)

    # Step 5: Figure Extraction (page-level)
    figures = extract_figures_from_pdf(pdf_path)

    # Step 6: OCR
    ocr_docs = ocr_images(figures)
    all_docs.extend(ocr_docs)

    # Step 7: Image Descriptions
    img_desc_docs = create_image_descriptions(
        figures,
        title=structure.get("title", ""),
        abstract=structure.get("abstract", ""),
    )
    all_docs.extend(img_desc_docs)

    # Step 8: Reference Extraction
    ref_docs = extract_references_from_pdf(pdf_path)
    all_docs.extend(ref_docs)

    # Summary
    logger.info("\n" + "-" * 40)
    logger.info("📊 DOCUMENT COLLECTION SUMMARY:")
    logger.info(f"   📝 Structured docs: {len(structured_docs)}")
    logger.info(f"   📄 Text pages: {len(text_docs)}")
    logger.info(f"   📊 Tables: {len(table_docs)}")
    logger.info(f"   🔍 OCR docs: {len(ocr_docs)}")
    logger.info(f"   🎨 Image descriptions: {len(img_desc_docs)}")
    logger.info(f"   📚 Reference docs: {len(ref_docs)}")
    logger.info(f"\n📚 Total: {len(all_docs)} documents")

    return all_docs
