import unicodedata
import re
from typing import List

def normalize(text: str, *, removeAccents: bool = True, keepApostrophes: bool = False):
    text = unicodedata.normalize("NFKC", text).casefold()
    
    # normalize quotes
    text = re.sub(r"\p{Pi}|\p{Pf}", "'", text)
    
    # turn dashes to spaces for words (pre-spark -> pre spark)
    text = re.sub(r"(?<=\w)\p{Pd}(?=\w)", " ", text)

    if removeAccents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

    if keepApostrophes:
        text = text.replace("'", " ")
    
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(cleanText: str) -> List[str]:
    return normalize(cleanText).split()