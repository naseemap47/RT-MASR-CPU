# benchmark/runners/accuracy_runner.py
"""
Accuracy runner: transcribes labelled audio and computes WER/CER.

Reference entries come from benchmark/data/references.yaml:
  - audio:  path to audio file (relative to project root)
  - lang:   "en" | "id" | "zh"
  - metric: "wer" | "cer"
  - text:   ground-truth transcript
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from benchmark.metrics.asr_metrics import normalise, wer, cer


@dataclass
class AccuracyResult:
    """Accuracy measurement for one audio file."""
    config_id: str
    audio_file: str
    lang: str
    metric: str          # "wer" or "cer"
    hypothesis: str      # raw model output
    reference: str       # raw reference text
    normalised_hyp: str
    normalised_ref: str
    score: float         # WER or CER value


class AccuracyRunner:
    """
    Transcribes each reference audio file once and computes WER/CER.

    Args:
        config_id:   Configuration identifier.
        engine:      ASR engine with `.transcribe(audio_path) -> dict` method.
        references:  List of reference dicts (loaded from references.yaml).
                     Each dict must have: audio, lang, metric, text.
    """

    def __init__(
        self,
        config_id: str,
        engine: Any,
        references: list[dict],
    ) -> None:
        self.config_id = config_id
        self.engine = engine
        self.references = references

    def run(self) -> list[AccuracyResult]:
        """
        Transcribe all reference files and compute error rates.

        Skips entries whose audio file does not exist (logs a warning).

        Returns:
            List of AccuracyResult — one per successfully processed file.
        """
        results: list[AccuracyResult] = []

        for ref in self.references:
            audio_file = ref.get("audio", "")
            lang = ref.get("lang", "en")
            metric = ref.get("metric", "wer")
            reference_text = ref.get("text", "")

            if not os.path.exists(audio_file):
                print(f"  [accuracy] SKIP missing: {audio_file}")
                continue

            print(f"  [accuracy] {audio_file} ({lang}, {metric})")
            try:
                result = self.engine.transcribe(audio_file)
                hypothesis = result.get("text", "")
            except Exception as exc:
                print(f"  [accuracy] ERROR on {audio_file}: {exc}")
                continue

            norm_hyp = normalise(hypothesis, lang)
            norm_ref = normalise(reference_text, lang)

            if metric == "cer":
                score = cer(hypothesis, reference_text)
            else:
                score = wer(hypothesis, reference_text, lang)

            results.append(AccuracyResult(
                config_id=self.config_id,
                audio_file=audio_file,
                lang=lang,
                metric=metric,
                hypothesis=hypothesis,
                reference=reference_text,
                normalised_hyp=norm_hyp,
                normalised_ref=norm_ref,
                score=score,
            ))

        return results
