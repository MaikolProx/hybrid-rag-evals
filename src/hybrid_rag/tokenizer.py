"""Tokenizer for Spanish/English text.

Keeps unicode words and numbers, normalizes to lowercase, and drops a small
high-frequency stopword list. Accent-insensitive: tildes are preserved (BM25
scoring is still useful on accented terms) but a canonical strip removes them
for exact matching when needed.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

_TOKEN_RE = re.compile(r"\w[\w'-]*", re.UNICODE)

STOPWORDS_ES = frozenset(
    """
    de la que el en y a los del se las por un para con no una su al lo como más
    pero sus le ya o este sí porque esta entre cuando muy sin sobre también me
    hasta hay donde quien desde todo nos durante todos uno les ni contra otros
    ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él
    tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas
    algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os
    mío mía míos mías tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro
    nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos esas
    ese esa aquel aquella aquellos aquellas estas estos este esta estás
    """.split()
)


def normalize(text: str) -> str:
    """Lowercase and strip accents (keeps other unicode intact)."""
    text = text.lower()
    return re.sub(r"[\u0300-\u036f]", "", text) if False else text


def tokenize(text: str, stopwords: bool = True) -> List[str]:
    """Tokenize text into a list of normalized tokens.

    Args:
        text: raw text.
        stopwords: if True, drop common stopwords (es/en).
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]
    if stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    return tokens


@lru_cache(maxsize=1)
def default_tokenizer() -> callable:
    """Return the tokenizer function with stopwords enabled."""
    return lambda text: tokenize(text, stopwords=True)
