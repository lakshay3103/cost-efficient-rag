"""
Loaders that turn raw files (PDF/HTML/MD) into plain text plus basic
per-document metadata (source path, title if we can find one).
"""
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import markdown as md_lib


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def load_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    html = md_lib.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


LOADERS = {
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".md": load_md,
    ".markdown": load_md,
}


def load_document(path: Path) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext} ({path})")
    text = LOADERS[ext](path)
    # Collapse excessive blank lines/whitespace from extraction noise.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
