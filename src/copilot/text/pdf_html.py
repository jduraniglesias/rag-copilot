from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader

from copilot.text.chunk import chunk_text

def ingest_pdf(path: str, size: int = 600, overlap: int = 120) -> List[Dict]:
    out: List[Dict] = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if not txt.strip():
            continue
        chunks = chunk_text(txt, doc_id=Path(path).name, size=size, overlap=overlap)
        for ch in chunks:
            ch["meta"]["page"] = i
        out.extend(chunks)
    return out

def ingest_html(path: str, size: int = 600, overlap: int = 120) -> List[Dict]:
    html = Path(path).read_text("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    chunks = chunk_text(text, doc_id=Path(path).name, size=size, overlap=overlap)
    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
    for ch in chunks:
        if title:
            ch["meta"]["title"] = title
    return chunks
