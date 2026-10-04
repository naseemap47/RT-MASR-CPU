# benchmark/metrics/asr_metrics.py
"""
ASR accuracy metrics: WER (English/Indonesian) and CER (Mandarin).

No external dependencies — stdlib only (unicodedata, re).
WER/CER use Wagner-Fischer edit distance (substitution=insertion=deletion=1).
"""
from __future__ import annotations

import re
import unicodedata

# CJK punctuation to strip for Mandarin normalisation
_ZH_PUNCT = (
    "\u3000\u3001\u3002\u300c\u300d\u300e\u300f\u3010\u3011"  # 　、。「」『』【】
    "\uff01\uff0c\uff0e\uff1a\uff1b\uff1f"                       # ！，．：；？
    "\u2018\u2019\u201c\u201d\u2026\u2014"                       # ''""…—
    "\u00b7"                                                      # ·
)
_ZH_PUNCT_SET = set(_ZH_PUNCT)


def edit_distance(a: list, b: list) -> int:
    """
    Wagner-Fischer edit distance between two sequences.
    Cost: substitution=1, insertion=1, deletion=1.
    """
    m, n = len(a), len(b)
    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalise(text: str, lang: str) -> str:
    """
    Normalise a transcript for error-rate computation.

    Args:
        text: Raw hypothesis or reference text.
        lang: Language code — "en", "id", or "zh".

    Returns:
        Normalised string.
    """
    # Step 1: Unicode NFKC (normalises full-width chars, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    if lang in ("en", "id"):
        # Lowercase
        text = text.lower()
        # Remove all characters that are not alphanumeric or space
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

    elif lang == "zh":
        # Remove CJK punctuation
        text = "".join(ch for ch in text if ch not in _ZH_PUNCT_SET)
        # Remove ASCII punctuation
        text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

    else:
        # Unknown language: NFKC + collapse whitespace only
        text = re.sub(r"\s+", " ", text).strip()

    return text


def wer(hypothesis: str, reference: str, lang: str) -> float:
    """
    Word Error Rate for English or Indonesian.

    Normalises both strings, splits on whitespace, then computes
    edit distance at the word level.

    Returns:
        WER in [0, ∞). Values > 1.0 possible when hypothesis is longer
        than reference. Returns 0.0 if both are empty.
    """
    hyp_norm = normalise(hypothesis, lang)
    ref_norm = normalise(reference, lang)

    hyp_words = hyp_norm.split() if hyp_norm else []
    ref_words = ref_norm.split() if ref_norm else []

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = edit_distance(hyp_words, ref_words)
    return dist / len(ref_words)


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate for Mandarin (or any character-level metric).

    Normalises both strings with lang="zh", then computes edit distance
    at the character level (excluding spaces).

    Returns:
        CER in [0, ∞). Returns 0.0 if both are empty.
    """
    hyp_norm = normalise(hypothesis, "zh").replace(" ", "")
    ref_norm = normalise(reference, "zh").replace(" ", "")

    hyp_chars = list(hyp_norm)
    ref_chars = list(ref_norm)

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = edit_distance(hyp_chars, ref_chars)
    return dist / len(ref_chars)
