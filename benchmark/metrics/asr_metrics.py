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
        # Apostrophes are deleted (not replaced) so "don't" == "dont" regardless of
        # which side kept the apostrophe.
        text = text.replace("'", "").replace("\u2019", "")
        # Every other non-word character (punctuation, hyphens, symbols) becomes a space.
        # \w is Unicode-aware, so accented letters (e.g. "café") are preserved.
        text = re.sub(r"[^\w\s]|_", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

    elif lang == "zh":
        # Latin letters inside Chinese text are case-folded
        text = text.lower()
        # Remove every punctuation (Unicode category P*) and symbol (S*) character.
        # This covers CJK marks (，。、《》「」…), ASCII punctuation and full-width forms.
        text = "".join(
            ch for ch in text
            if ch not in _ZH_PUNCT_SET
            and not unicodedata.category(ch).startswith(("P", "S"))
        )
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

    else:
        # Unknown language: NFKC + collapse whitespace only
        text = re.sub(r"\s+", " ", text).strip()

    return text


def wer_counts(hypothesis: str, reference: str, lang: str) -> tuple[int, int]:
    """
    Word-level (edit_distance, reference_length) after normalisation.

    The pair is what you need for corpus-level WER: sum(edits) / sum(ref_len).
    """
    hyp_words = normalise(hypothesis, lang).split()
    ref_words = normalise(reference, lang).split()
    return edit_distance(hyp_words, ref_words), len(ref_words)


def cer_counts(hypothesis: str, reference: str, lang: str = "zh") -> tuple[int, int]:
    """Character-level (edit_distance, reference_length); spaces are ignored."""
    hyp_chars = list(normalise(hypothesis, lang).replace(" ", ""))
    ref_chars = list(normalise(reference, lang).replace(" ", ""))
    return edit_distance(hyp_chars, ref_chars), len(ref_chars)


def _rate(edits: int, ref_len: int, hyp_nonempty: bool) -> float:
    if ref_len == 0:
        return 1.0 if hyp_nonempty else 0.0
    return edits / ref_len


def wer(hypothesis: str, reference: str, lang: str) -> float:
    """
    Word Error Rate (substitutions + insertions + deletions) / reference words.

    Both strings are normalised first. Values > 1.0 are possible when the
    hypothesis is much longer than the reference. Empty reference: 0.0 if the
    hypothesis is also empty, else 1.0.
    """
    edits, n = wer_counts(hypothesis, reference, lang)
    return _rate(edits, n, bool(normalise(hypothesis, lang)))


def cer(hypothesis: str, reference: str, lang: str = "zh") -> float:
    """
    Character Error Rate for Mandarin (or any character-level metric).

    Same definition as :func:`wer` but over characters (spaces ignored).
    """
    edits, n = cer_counts(hypothesis, reference, lang)
    return _rate(edits, n, bool(normalise(hypothesis, lang).replace(" ", "")))
