from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader

from copilot.text.chunk import chunk_text

def ingest_pdf(path: str, size=600, overlap=120) -> List[Dict]:
    out = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if not txt.strip(): continue
        chunks = chunk_text(txt, doc_id=Path(path).name, size=size, overlap=overlap)
        # add page metadata to each chunk
        for ch in chunks:
            ch["meta"]["page"] = i
        out.extend(chunks)
    return out

def ingest_html(path_or_str: str, size=600, overlap=120, from_string=False) -> List[Dict]:
    html = path_or_str if from_string else Path(path_or_str).read_text("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    # basic readability-ish extraction
    for tag in soup(["script","style","noscript"]): tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    chunks = chunk_text(text, doc_id=Path(path_or_str).name if not from_string else "page.html",
                        size=size, overlap=overlap)
    # optional: store <title> or URL
    title = (soup.title.string.strip() if soup.title else None)
    for ch in chunks:
        if title: ch["meta"]["title"] = title
    return chunks
