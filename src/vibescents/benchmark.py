from __future__ import annotations

import time
from statistics import mean
from typing import Iterable

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
"""


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


class BenchmarkGenerator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API.")

        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    def generate_case_labels(
        self,
        *,
        case_id: str,
        brief: str,
        runs: int = 3,
    ) -> list[BenchmarkCaseDraft]:
        from google.genai import types

        drafts: list[BenchmarkCaseDraft] = []
        for run_idx in range(runs):
            for attempt in range(5):
                try:
                    response = self._client.models.generate_content(
                        model=self.settings.reranker_model,
                        contents=f"case_id: {case_id}\nbrief: {brief}",
                        config=types.GenerateContentConfig(
                            system_instruction=BENCHMARK_PROMPT,
                            response_mime_type="application/json",
                            response_schema=BenchmarkCaseLabel,
                        ),
                    )
                    parsed = getattr(response, "parsed", None)
                    label = (
                        parsed
                        if isinstance(parsed, BenchmarkCaseLabel)
                        else BenchmarkCaseLabel.model_validate(parsed)
                        if parsed is not None
                        else BenchmarkCaseLabel.model_validate_json(response.text)
                    )
                    drafts.append(BenchmarkCaseDraft(case_id=case_id, brief=brief, labels=label))
                    break
                except Exception as e:
                    if attempt < 4 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                        wait = min(30 * (attempt + 1), 120)
                        print(f"  Rate limited on {case_id} run {run_idx + 1}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            time.sleep(2)
        return drafts
