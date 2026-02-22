"""
Text cleaning utilities extracted from the enhanced notebook (Cell 7).
All regex patterns and cleaning logic preserved exactly.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF.
    Preserves paragraph structure while removing artifacts.

    From notebook Cell 7 — unchanged.
    """
    if not text:
        return ""

    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers (standalone numbers on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove header/footer artifacts (common in academic papers)
    text = re.sub(r'^.*(?:Published|Preprint|Draft|Confidential).*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Normalize whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Clean up resulting empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into single space."""
    return re.sub(r'[ \t]+', ' ', text).strip()


def remove_hyphenation(text: str) -> str:
    """Remove end-of-line hyphenation common in PDFs."""
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)


def is_likely_header_footer(text: str) -> bool:
    """
    Check if a text block is likely a header or footer.
    Used to filter out noise from PDF extraction.
    """
    if not text or len(text.strip()) < 5:
        return True

    # Very short lines that are just numbers (page numbers)
    if re.match(r'^\s*\d{1,4}\s*$', text.strip()):
        return True

    # Common header/footer patterns
    header_patterns = [
        r'^.*(?:page|pg\.?)\s*\d+.*$',
        r'^\s*\d+\s*of\s*\d+\s*$',
    ]
    for pattern in header_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True

    return False
