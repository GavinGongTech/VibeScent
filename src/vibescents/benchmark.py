from __future__ import annotations

import json
import time
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Protocol

from pydantic import ValidationError

from vibescents.schemas import BenchmarkCaseDraft, BenchmarkCaseLabel
from vibescents.settings import Settings
from vibescents.similarity import frequent_items, majority_vote

BENCHMARK_PROMPT = """\
Expand the benchmark brief into structured fragrance-target labels.
Use category labels that help fragrance retrieval:
- formality
- season
- day vs night
- fresh vs warm
- acceptable accords
- acceptable note families
- disallowed traits
- example good fragrances
- confidence in [0, 1]
"""

GEMINI_BENCHMARK_MODEL = "gemini-3.1-pro-preview"
QWEN_BENCHMARK_MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"


def consolidate_case_drafts(drafts: Iterable[BenchmarkCaseDraft]) -> BenchmarkCaseLabel:
    draft_list = list(drafts)
    if not draft_list:
        raise ValueError("At least one draft is required to consolidate benchmark labels.")
    if len({draft.case_id for draft in draft_list}) != 1:
        raise ValueError("All drafts must refer to the same case_id.")

    labels = [draft.labels for draft in draft_list]
    return BenchmarkCaseLabel(
        case_id=labels[0].case_id,
        occasion_text=majority_vote([label.occasion_text for label in labels]),
        target_formality=majority_vote([label.target_formality for label in labels]),
        target_season=majority_vote([label.target_season for label in labels]),
        target_day_night=majority_vote([label.target_day_night for label in labels]),
        target_fresh_warm=majority_vote([label.target_fresh_warm for label in labels]),
        acceptable_accords=frequent_items(label.acceptable_accords for label in labels),
        acceptable_note_families=frequent_items(label.acceptable_note_families for label in labels),
        disallowed_traits=frequent_items(label.disallowed_traits for label in labels),
        example_good_fragrances=frequent_items(label.example_good_fragrances for label in labels),
        confidence=round(mean(label.confidence for label in labels), 3),
    )


class BenchmarkLabelClient(Protocol):
    def generate(self, *, case_id: str, brief: str) -> BenchmarkCaseLabel:
        """Generate a benchmark label for one case."""


@dataclass
class GeminiBenchmarkLabelClient:
    model_name: str = GEMINI_BENCHMARK_MODEL
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API.")
        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    def generate(self, *, case_id: str, brief: str) -> BenchmarkCaseLabel:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=f"case_id: {case_id}\nbrief: {brief}",
            config=types.GenerateContentConfig(
                system_instruction=BENCHMARK_PROMPT,
                response_mime_type="application/json",
                response_schema=BenchmarkCaseLabel,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BenchmarkCaseLabel):
            return parsed
        if parsed is not None:
            return BenchmarkCaseLabel.model_validate(parsed)
        return BenchmarkCaseLabel.model_validate_json(response.text)


@dataclass
class QwenOutlinesBenchmarkLabelClient:
    model_name: str = QWEN_BENCHMARK_MODEL

    def __post_init__(self) -> None:
        self._generator = _build_outlines_generator(self.model_name)

    def generate(self, *, case_id: str, brief: str) -> BenchmarkCaseLabel:
        prompt = f"case_id: {case_id}\nbrief: {brief}"
        raw = self._generator(f"{BENCHMARK_PROMPT}\n\n{prompt}")
        parsed = _parse_benchmark_label(raw)
        if parsed is not None:
            return parsed
        repaired = _repair_payload(raw)
        parsed = _parse_benchmark_label(repaired)
        if parsed is not None:
            return parsed
        raise ValueError("Outlines output could not be parsed into BenchmarkCaseLabel.")


def _build_outlines_generator(model_name: str):
    try:
        import outlines
    except ImportError as exc:
        raise ImportError(
            "Qwen provider requires `outlines`. Install it in Colab with notebooks/requirements.colab.txt."
        ) from exc
    try:
        model = outlines.models.vllm(model_name)
    except Exception:
        model = outlines.models.transformers(model_name, device="cuda")
    return outlines.generate.json(model, BenchmarkCaseLabel)


def _repair_payload(payload: object) -> object:
    try:
        from json_repair import repair_json
    except ImportError:
        return payload
    return repair_json(str(payload))


def _parse_benchmark_label(payload: object) -> BenchmarkCaseLabel | None:
    try:
        if isinstance(payload, BenchmarkCaseLabel):
            return payload
        if isinstance(payload, dict):
            return BenchmarkCaseLabel.model_validate(payload)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return BenchmarkCaseLabel.model_validate(parsed)
        return None
    except (ValidationError, TypeError, ValueError):
        return None


class BenchmarkGenerator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: str = "qwen",
        model_name: str | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.provider = provider
        self.model_name = model_name
        self._client = self._resolve_client()

    def _resolve_client(self) -> BenchmarkLabelClient:
        if self.provider == "gemini":
            return GeminiBenchmarkLabelClient(
                model_name=self.model_name or self.settings.reranker_model or GEMINI_BENCHMARK_MODEL,
                settings=self.settings,
            )
        if self.provider == "qwen":
            return QwenOutlinesBenchmarkLabelClient(model_name=self.model_name or QWEN_BENCHMARK_MODEL)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_case_labels(
        self,
        *,
        case_id: str,
        brief: str,
        runs: int = 3,
        confidence_threshold: float = 0.6,
        adaptive_reruns: int = 2,
    ) -> list[BenchmarkCaseDraft]:
        drafts: list[BenchmarkCaseDraft] = []
        total_runs = runs

        while len(drafts) < total_runs:
            for attempt in range(5):
                try:
                    label = self._client.generate(case_id=case_id, brief=brief)
                    drafts.append(BenchmarkCaseDraft(case_id=case_id, brief=brief, labels=label))
                    break
                except Exception as exc:
                    retryable = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                    if retryable and attempt < 4:
                        delay = min(60.0, 2.0 ** (attempt + 1))
                        time.sleep(delay)
                        continue
                    raise
            time.sleep(1)

            if (
                len(drafts) == runs
                and adaptive_reruns > 0
                and consolidate_case_drafts(drafts).confidence < confidence_threshold
            ):
                total_runs = runs + adaptive_reruns

        return drafts
