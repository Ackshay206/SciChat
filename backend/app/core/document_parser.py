"""
Document parsing module — ALL functions extracted verbatim from the enhanced notebook (Cells 7-8).

Contains:
- Structure analysis (title, authors, emails, organizations, abstract, section headings)
- Structured text extraction (metadata, abstract, sections as separate docs)
- Full text extraction (page-by-page)
- Reference extraction and parsing
- Table extraction (hybrid 4-method approach)
- Image extraction + OCR (EasyOCR)
- Image description generation (Gemini Vision)
- Complete document collection pipeline
"""

import io
import re
import logging
from typing import Any, Dict, List

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber
import tabula
from PIL import Image
from google import genai
from google.genai import types
from llama_index.core.schema import Document

from app.config import config
from app.utils.text_cleaning import clean_text

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURE ANALYSIS
# =============================================================================

def analyze_document_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF structure to extract:
    - Title
    - Authors
    - Emails
    - Organizations/Affiliations
    - Abstract
    - Section headings

    From notebook Cell 7 — unchanged.
    """
    logger.info("📋 Analyzing document structure...")

    structure = {
        "title": "",
        "authors": [],
        "emails": [],
        "organizations": [],
        "abstract": "",
        "sections": [],
        "raw_first_pages": ""
    }

    doc = fitz.open(pdf_path)

    # Extract text from first 2-3 pages for header analysis
    header_text = ""
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        header_text += page.get_text("text") + "\n"

    structure["raw_first_pages"] = header_text

    # Extract title (usually the largest font on first page)
    structure["title"] = extract_title(doc, header_text)

    # Extract authors
    structure["authors"] = extract_authors(header_text)

    # Extract emails
    structure["emails"] = extract_emails(header_text)

    # Extract organizations/affiliations
    structure["organizations"] = extract_organizations(header_text)

    # Extract abstract
    structure["abstract"] = extract_abstract(header_text)

    # Extract section headings from entire document
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text("text") + "\n"

    structure["sections"] = extract_section_headings(full_text)

    doc.close()

    # Log summary
    logger.info(f"   📌 Title: {structure['title'][:60]}..." if len(structure['title']) > 60 else f"   📌 Title: {structure['title']}")
    logger.info(f"   👥 Authors: {len(structure['authors'])} found")
    logger.info(f"   📧 Emails: {len(structure['emails'])} found")
    logger.info(f"   🏢 Organizations: {len(structure['organizations'])} found")
    logger.info(f"   📝 Abstract: {'Found' if structure['abstract'] else 'Not found'}")
    logger.info(f"   📑 Sections: {len(structure['sections'])} found")

    return structure


def extract_title(doc, header_text: str) -> str:
    """Extract paper title using font analysis and heuristics."""
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    title_candidates = []

    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size = span["size"]
                    text = span["text"].strip()

                    # Title is usually large font (>14pt) and near top
                    if font_size > 14 and len(text) > 10:
                        title_candidates.append((font_size, text))

    # Sort by font size (largest first)
    title_candidates.sort(key=lambda x: x[0], reverse=True)

    if title_candidates:
        title = title_candidates[0][1]
        return clean_text(title)

    # Fallback: Use first substantial line
    lines = header_text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not re.match(r'^(Abstract|Introduction|\d+\.)', line):
            return clean_text(line)

    return ""


def extract_authors(header_text: str) -> List[str]:
    """Extract author names from header text."""
    authors = []

    lines = header_text.split('\n')

    # Look for author block (usually after title, before abstract)
    in_author_section = False
    author_lines = []

    for i, line in enumerate(lines[:30]):  # Check first 30 lines
        line = line.strip()

        # Skip empty lines and common non-author content
        if not line:
            continue
        if re.match(r'^(Abstract|Introduction|\d+\.|Keywords|arXiv)', line, re.IGNORECASE):
            in_author_section = False
            continue

        # Look for lines with multiple capitalized words (potential names)
        if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line):
            words = line.split()
            capitalized_ratio = sum(1 for w in words if w[0].isupper()) / len(words) if words else 0

            if capitalized_ratio > 0.5 and len(line) < 200:
                # Remove affiliations markers (superscripts, asterisks)
                cleaned = re.sub(r'[∗†‡§¶\d,]+', ' ', line)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                # Split by common separators
                potential_names = re.split(r'\s+and\s+|,\s*', cleaned)

                for name in potential_names:
                    name = name.strip()
                    name_parts = name.split()
                    if 2 <= len(name_parts) <= 4 and all(p[0].isupper() for p in name_parts if p):
                        if name not in authors:
                            authors.append(name)

    return authors[:20]  # Limit to 20 authors


def extract_emails(header_text: str) -> List[str]:
    """Extract email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, header_text)
    return list(set(emails))


def extract_organizations(header_text: str) -> List[str]:
    """Extract organizations/affiliations from header text."""
    organizations = []

    # Common organization keywords
    org_keywords = [
        r'University', r'Institute', r'Laboratory', r'Lab',
        r'Department', r'School', r'College', r'Center',
        r'Corporation', r'Inc\.', r'Ltd\.', r'Research',
        r'Google', r'Microsoft', r'Meta', r'OpenAI', r'DeepMind',
        r'Facebook', r'Amazon', r'IBM', r'NVIDIA'
    ]

    pattern = r'(' + '|'.join(org_keywords) + r')[^,\n]*'

    matches = re.findall(pattern, header_text, re.IGNORECASE)

    # Also look for lines that seem like affiliations
    lines = header_text.split('\n')
    for line in lines[:40]:
        line = line.strip()
        for keyword in org_keywords:
            if re.search(keyword, line, re.IGNORECASE):
                cleaned = re.sub(r'[∗†‡§¶\d]+', '', line).strip()
                if len(cleaned) > 10 and cleaned not in organizations:
                    organizations.append(cleaned)
                break

    return organizations[:10]  # Limit to 10


def extract_abstract(header_text: str) -> str:
    """Extract the abstract section."""
    abstract_patterns = [
        r'Abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1\s*[\.\s]|Introduction|Keywords)',
        r'ABSTRACT\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1\s*[\.\s]|INTRODUCTION|Keywords)',
    ]

    for pattern in abstract_patterns:
        match = re.search(pattern, header_text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]

    return ""


def extract_section_headings(full_text: str) -> List[Dict[str, Any]]:
    """
    Extract section headings from the document.
    - Handles split headers (number on one line, title on next)
    - Filters out years, decimals, and reference entries
    - Keeps table captions as headings

    From notebook Cell 7 — unchanged.
    """

    acronym_exclusions = {
        'BLEU', 'GPU', 'CPU', 'LSTM', 'RNN', 'CNN', 'NLP', 'WMT', 'TPU',
        'ADAM', 'SGD', 'RELU', 'GELU', 'FFN', 'MLP', 'BERT', 'GPT', 'LLM',
        'PPL', 'GNMT', 'API', 'URL', 'HTTP', 'HTML', 'JSON', 'XML', 'SQL'
    }

    keyword_headings = {
        "abstract", "acknowledgements", "acknowledgments", "references",
        "appendix", "appendices"
    }

    def is_number_only(s: str) -> bool:
        return re.match(r"^\d+(?:\.\d+)*\.?$", s) is not None

    def is_valid_section_number(num: str) -> bool:
        if not re.match(r"^\d+(?:\.\d+)*$", num):
            return False
        parts = num.split(".")
        if len(parts) > 4:
            return False
        if len(parts[0]) > 2:  # reject 2014, 106, etc.
            return False
        if any(len(p) > 2 for p in parts[1:]):  # reject 3.200, 3.2.100
            return False
        return True

    def looks_like_title(title: str) -> bool:
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

    def is_reference_entry(s: str) -> bool:
        return re.match(r"^\[\d+\]\s+", s) is not None

    def is_table_caption(s: str) -> bool:
        return re.match(r"^(Table|TABLE)\s+\d+\s*:\s+.+$", s) is not None

    # Normalize lines
    lines_raw = full_text.splitlines()
    lines = []
    for ln in lines_raw:
        s = re.sub(r"\s+", " ", (ln or "")).strip()
        if s:
            lines.append(s)

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

        # 1) Table captions
        if is_table_caption(s):
            m = re.match(r"^(Table|TABLE)\s+(?P<num>\d+)\s*:\s*(?P<title>.+)$", s)
            sections.append({
                "heading": s,
                "line_number": i,
                "number": f"Table {m.group('num')}",
                "title": f"Table {m.group('num')}: {m.group('title').strip()}",
                "pattern": "table_caption"
            })
            i += 1
            continue

        # 2) Merge split headers: "3.2.1" + next line title
        if is_number_only(s) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if looks_like_title(nxt):
                s = f"{s.rstrip('.')} {nxt}"
                i += 1

        # 3) Skip reference entries
        if is_reference_entry(s):
            i += 1
            continue

        # 4) Keyword headings (Abstract, References, etc.)
        if s.strip().lower() in keyword_headings:
            title = s.strip()
            sections.append({
                "heading": title,
                "line_number": i,
                "number": None,
                "title": title,
                "pattern": "keyword_heading"
            })
            if title.lower() == "references":
                in_references = True
            i += 1
            continue

        # 5) ALL CAPS headings
        m_caps = re.match(r"^([A-Z]{3,}(?:\s+[A-Z]{3,})*)$", s)
        if m_caps:
            heading = m_caps.group(1)
            if heading not in acronym_exclusions and len(heading) >= 5:
                sections.append({
                    "heading": heading,
                    "line_number": i,
                    "number": None,
                    "title": heading,
                    "pattern": "all_caps"
                })
                if heading.lower() == "references":
                    in_references = True
                i += 1
                continue

        # 6) Numbered headings: "1 Introduction", "6.1 Machine Translation"
        m_num = re.match(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>.+)$", s)
        if m_num:
            num = m_num.group("num")
            title = m_num.group("title").strip()

            if is_valid_section_number(num) and looks_like_title(title):
                sections.append({
                    "heading": f"{num} {title}",
                    "line_number": i,
                    "number": num,
                    "title": title,
                    "pattern": "numbered"
                })
                i += 1
                continue

        i += 1

    return sections


# =============================================================================
# TEXT EXTRACTION WITH STRUCTURE ANALYSIS
# =============================================================================

def extract_structured_text_from_pdf(pdf_path: str, structure: Dict[str, Any]) -> List[Document]:
    """
    Extract text with structure-aware metadata.
    Creates separate documents for metadata, abstract, and each major section.

    From notebook Cell 7 — unchanged.
    """
    logger.info("📄 Extracting structured text...")
    documents = []

    # Create metadata document
    if structure["title"] or structure["authors"]:
        metadata_text = f"## Paper Metadata\n\n"
        if structure["title"]:
            metadata_text += f"**Title:** {structure['title']}\n\n"
        if structure["authors"]:
            metadata_text += f"**Authors:** {', '.join(structure['authors'])}\n\n"
        if structure["emails"]:
            metadata_text += f"**Contact:** {', '.join(structure['emails'])}\n\n"
        if structure["organizations"]:
            metadata_text += f"**Affiliations:** {'; '.join(structure['organizations'])}\n\n"

        documents.append(Document(
            text=metadata_text,
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
        abstract_text = f"## Abstract\n\n{structure['abstract']}"
        documents.append(Document(
            text=abstract_text,
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
        page = doc[page_num]
        full_text += page.get_text("text") + "\n\n"
    doc.close()

    # Split by detected sections
    if structure["sections"]:
        sections_with_content = extract_sections_content(full_text, structure["sections"])

        for section in sections_with_content:
            if section["content"] and len(section["content"]) > 50:
                section_text = f"## {section['heading']}\n\n{section['content']}"
                documents.append(Document(
                    text=clean_text(section_text),
                    metadata={
                        "source": pdf_path,
                        "type": "section",
                        "content_type": "section",
                        "section_name": section.get("title", section["heading"]),
                        "section_number": section.get("number", ""),
                    }
                ))

    logger.info(f"   ✅ {len(documents)} structured documents created")
    return documents


def extract_sections_content(full_text: str, sections: List[Dict]) -> List[Dict]:
    """Extract content for each section."""
    lines = full_text.split('\n')
    sections_with_content = []

    for i, section in enumerate(sections):
        start_line = section["line_number"]

        # Find end (next section or end of document)
        if i + 1 < len(sections):
            end_line = sections[i + 1]["line_number"]
        else:
            end_line = len(lines)

        # Extract content
        content_lines = lines[start_line + 1:end_line]
        content = '\n'.join(content_lines)

        sections_with_content.append({
            **section,
            "content": content.strip()
        })

    return sections_with_content


# =============================================================================
# FULL TEXT EXTRACTION (ORIGINAL APPROACH - PRESERVED)
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """Extract text from PDF using PyMuPDF (original approach)."""
    logger.info("📄 Extracting full text (page by page)...")
    documents = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

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
# REFERENCE EXTRACTION
# =============================================================================

def extract_references_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extract references/bibliography section from PDF.
    Creates separate documents for references.

    From notebook Cell 7 — unchanged.
    """
    logger.info("📚 Extracting references...")

    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text("text") + "\n"

    doc.close()

    # Find references section
    ref_patterns = [
        r'(?:^|\n)\s*References?\s*\n',
        r'(?:^|\n)\s*REFERENCES?\s*\n',
        r'(?:^|\n)\s*Bibliography\s*\n',
        r'(?:^|\n)\s*BIBLIOGRAPHY\s*\n',
    ]

    ref_start = -1
    for pattern in ref_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            ref_start = match.end()
            break

    if ref_start == -1:
        logger.info("   ⚠️ No references section found")
        return []

    # Extract reference text
    ref_text = full_text[ref_start:]

    # Try to find where references end
    end_patterns = [
        r'\n\s*(?:Appendix|APPENDIX|Supplementary|SUPPLEMENTARY)',
    ]

    for pattern in end_patterns:
        match = re.search(pattern, ref_text)
        if match:
            ref_text = ref_text[:match.start()]
            break

    # Parse individual references
    references = parse_references(ref_text)

    if not references:
        if ref_text.strip():
            return [Document(
                text=f"## References\n\n{clean_text(ref_text)}",
                metadata={
                    "source": pdf_path,
                    "type": "references",
                    "content_type": "references",
                    "reference_count": 0,
                }
            )]
        return []

    # Create reference documents — group into chunks
    documents = []

    ref_chunks = []
    current_chunk = []
    current_length = 0

    for ref in references:
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

    logger.info(f"   ✅ {len(references)} references extracted into {len(documents)} documents")
    return documents


def parse_references(ref_text: str) -> List[Dict[str, str]]:
    """Parse individual references from text."""
    references = []

    # Try numbered references first: [1] Author, Title...
    numbered_pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)'
    matches = re.findall(numbered_pattern, ref_text, re.DOTALL)

    if matches:
        for num, text in matches:
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 20:
                references.append({"number": num, "text": text})
        return references

    # Try period-numbered references: 1. Author, Title...
    period_pattern = r'(\d+)\.\s*(.*?)(?=\d+\.\s|\Z)'
    matches = re.findall(period_pattern, ref_text, re.DOTALL)

    if matches:
        for num, text in matches:
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 20:
                references.append({"number": num, "text": text})
        return references

    # Fallback: Split by blank lines
    lines = ref_text.split('\n')
    current_ref = ""
    ref_num = 1

    for line in lines:
        line = line.strip()
        if not line:
            if current_ref:
                references.append({"number": str(ref_num), "text": current_ref})
                ref_num += 1
                current_ref = ""
        else:
            current_ref += " " + line if current_ref else line

    if current_ref:
        references.append({"number": str(ref_num), "text": current_ref})

    return references


# =============================================================================
# TABLE EXTRACTION (HYBRID 4-METHOD APPROACH — UNCHANGED)
# =============================================================================

def is_reference_table(text: str) -> bool:
    """
    Check if the extracted table is actually a references/bibliography section.
    Returns True if it looks like references (should be filtered out).
    """
    if not text.strip():
        return False

    reference_patterns = [
        r'arXiv preprint',
        r'arXiv:\d+\.\d+',
        r'In Proceedings',
        r'Journal of',
        r'et al\.',
        r'pages \d+[–-]\d+',
        r'In Advances in Neural',
        r'preprint arXiv',
    ]

    lines = text.split('\n')
    reference_matches = 0
    total_lines = 0

    for line in lines:
        line = line.strip()
        if not line or line == 'nan':
            continue
        total_lines += 1

        for pattern in reference_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                reference_matches += 1
                break

    if total_lines > 0 and reference_matches / total_lines > 0.3:
        return True

    return False


def extract_tables_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extract tables using hybrid approach (UNCHANGED):
    1. Tabula (Lattice mode) - for bordered tables
    2. Tabula (Stream mode) - for borderless tables
    3. PDFPlumber - alternative extraction
    4. PyMuPDF - built-in table detection
    """
    logger.info("📊 Extracting tables (hybrid approach)...")

    all_table_texts = []
    references_filtered = 0

    # METHOD 1: TABULA LATTICE
    logger.info("   🔷 Tabula Lattice...")
    try:
        tables_lattice = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            lattice=True,
            silent=True
        )

        for i, df in enumerate(tables_lattice):
            if not df.empty:
                table_text = df.to_markdown(index=False)

                if is_reference_table(table_text):
                    references_filtered += 1
                    continue

                if table_text.strip():
                    all_table_texts.append(f"[Tabula-Lattice Table {i+1}]\n{table_text}")

        logger.info(f"      Found {len(tables_lattice)} tables")
    except Exception as e:
        logger.warning(f"      Tabula Lattice error: {e}")

    # METHOD 2: TABULA STREAM
    logger.info("   🔶 Tabula Stream...")
    try:
        tables_stream = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            stream=True,
            silent=True
        )

        for i, df in enumerate(tables_stream):
            if not df.empty:
                table_text = df.to_markdown(index=False)

                if is_reference_table(table_text):
                    references_filtered += 1
                    continue

                if table_text.strip():
                    all_table_texts.append(f"[Tabula-Stream Table {i+1}]\n{table_text}")

        logger.info(f"      Found {len(tables_stream)} tables")
    except Exception as e:
        logger.warning(f"      Tabula Stream error: {e}")

    # METHOD 3: PDFPLUMBER
    logger.info("   🔷 PDFPlumber...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            plumber_count = 0
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        table_text = df.to_markdown(index=False)

                        if is_reference_table(table_text):
                            references_filtered += 1
                            continue

                        if table_text.strip():
                            all_table_texts.append(f"[PDFPlumber Page {page_num+1} Table {i+1}]\n{table_text}")
                            plumber_count += 1
            logger.info(f"      Found {plumber_count} tables")
    except Exception as e:
        logger.warning(f"      PDFPlumber error: {e}")

    # METHOD 4: PYMUPDF
    logger.info("   🔶 PyMuPDF...")
    try:
        doc = fitz.open(pdf_path)
        pymupdf_count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            for i, table in enumerate(tables):
                df = table.to_pandas()
                if not df.empty:
                    table_text = df.to_markdown(index=False)

                    if is_reference_table(table_text):
                        references_filtered += 1
                        continue

                    if table_text.strip():
                        all_table_texts.append(f"[PyMuPDF Page {page_num+1} Table {i+1}]\n{table_text}")
                        pymupdf_count += 1
        doc.close()
        logger.info(f"      Found {pymupdf_count} tables")
    except Exception as e:
        logger.warning(f"      PyMuPDF error: {e}")

    # Create documents
    table_docs = []
    for i, text in enumerate(all_table_texts):
        table_docs.append(Document(
            text=text,
            metadata={
                "type": "table",
                "content_type": "table",
                "table_index": i + 1
            }
        ))

    logger.info(f"   📋 References filtered: {references_filtered}")
    logger.info(f"   ✅ Total: {len(table_docs)} table documents")

    return table_docs


# =============================================================================
# IMAGE EXTRACTION + OCR + VISION DESCRIPTIONS (UNCHANGED)
# =============================================================================

def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract images from PDF."""
    logger.info("🖼️ Extracting images...")
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.width < 100 or pil_image.height < 100:
                    continue

                images.append({
                    "page": page_num + 1,
                    "index": img_idx + 1,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "image": pil_image,
                    "bytes": image_bytes,
                })
            except Exception:
                pass

    doc.close()
    logger.info(f"   ✅ {len(images)} images extracted")
    return images


def ocr_images(images: List[Dict]) -> List[Document]:
    """Run OCR on images using EasyOCR."""
    if not images:
        return []

    logger.info("🔍 Running OCR...")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    ocr_docs = []
    for img_data in images:
        try:
            image = img_data["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')

            results = reader.readtext(np.array(image))
            text = " ".join([r[1] for r in results])

            if text.strip():
                ocr_docs.append(Document(
                    text=f"Figure (page {img_data['page']}): {clean_text(text)}",
                    metadata={
                        "type": "image_ocr",
                        "content_type": "figure",
                        "page": img_data["page"]
                    }
                ))
        except Exception:
            pass

    logger.info(f"   ✅ OCR completed for {len(ocr_docs)} images")
    return ocr_docs


def create_image_descriptions(images: List[Dict]) -> List[Document]:
    """Create text descriptions for images using Gemini Vision."""
    if not images:
        return []

    logger.info("🎨 Creating image descriptions with Vision AI...")

    try:
        vision_client = genai.Client(api_key=config.GOOGLE_API_KEY)

        docs = []
        for idx, img_data in enumerate(images, 1):
            try:
                logger.info(f"   Processing {idx}/{len(images)} (page {img_data['page']})...")

                img_buffer = io.BytesIO()
                img_data["image"].save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()

                prompt = """Analyze this scientific figure/diagram and provide a detailed description.

Focus on:
1. Figure number and Type of diagram (architecture, flowchart, graph, table, etc.)
2. IMPORTANT: Identify the number of independent diagrams and their positions in the image if multiple are present.
3. Main components and their relationships.
4. Any text labels, equations, or annotations.
5. Key insights or patterns shown.

Be specific and technical. This description will be used for RAG retrieval."""

                response = vision_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                                types.Part.from_text(text=prompt),
                            ],
                        ),
                    ],
                )

                description = response.text

                if description:
                    docs.append(Document(
                        text=f"Figure from page {img_data['page']}: {description}",
                        metadata={
                            "type": "image_description",
                            "content_type": "figure",
                            "page": img_data["page"]
                        }
                    ))
                    logger.info(f"   ✅ Page {img_data['page']} described")
                else:
                    logger.warning(f"   ⚠️ No description for page {img_data['page']}")

            except Exception as e:
                logger.error(f"   ❌ Error on page {img_data['page']}: {str(e)[:50]}")

        logger.info(f"   ✅ {len(docs)} descriptions created")
        return docs

    except Exception as e:
        logger.error(f"   ❌ Vision API error: {e}")
        return []


# =============================================================================
# COMPLETE DOCUMENT COLLECTION PIPELINE (Cell 8 — unchanged)
# =============================================================================

def collect_all_documents(pdf_path: str) -> List[Document]:
    """
    Complete document collection pipeline with structure analysis.

    Flow:
    1. Structure Analysis (title, authors, emails, organizations, abstract, sections)
    2. Structured Text Extraction (metadata, abstract, sections as separate docs)
    3. Full Text Extraction (original page-by-page approach)
    4. Table Extraction (unchanged hybrid approach)
    5. Image Extraction + OCR (unchanged)
    6. Image Descriptions (unchanged)
    7. Reference Extraction
    """
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENT COLLECTION PIPELINE (ENHANCED)")
    logger.info("=" * 60)

    all_docs = []

    # Step 1: Structure Analysis
    structure = analyze_document_structure(pdf_path)

    # Step 2: Structured Text Extraction (with metadata)
    structured_docs = extract_structured_text_from_pdf(pdf_path, structure)
    all_docs.extend(structured_docs)

    # Step 3: Full Text Extraction (original approach)
    text_docs = extract_text_from_pdf(pdf_path)
    all_docs.extend(text_docs)

    # Step 4: Table Extraction (unchanged)
    table_docs = extract_tables_from_pdf(pdf_path)
    all_docs.extend(table_docs)

    # Step 5: Image Extraction
    images = extract_images_from_pdf(pdf_path)

    # Step 6: OCR (unchanged)
    ocr_docs = ocr_images(images)
    all_docs.extend(ocr_docs)

    # Step 7: Image Descriptions (unchanged)
    img_desc_docs = create_image_descriptions(images)
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
