from __future__ import annotations

from vibescents.benchmark import consolidate_case_drafts
from vibescents.schemas import BenchmarkCaseDraft, BenchmarkCaseLabel


def _draft(season: str, accords: list[str], confidence: float) -> BenchmarkCaseDraft:
    return BenchmarkCaseDraft(
        case_id="case_001",
        brief="Formal winter event",
        labels=BenchmarkCaseLabel(
            case_id="case_001",
            occasion_text="Formal winter event",
            target_formality="formal",
            target_season=season,
            target_day_night="night",
            target_fresh_warm="warm",
            acceptable_accords=accords,
            acceptable_note_families=["woody", "amber"],
            disallowed_traits=["overly sweet"],
            example_good_fragrances=["Example A", "Example B"],
            confidence=confidence,
        ),
    )


def test_consolidate_case_drafts_majority_votes_and_merges_lists() -> None:
    consolidated = consolidate_case_drafts(
        [
            _draft("winter", ["woody", "amber"], 0.8),
            _draft("winter", ["amber", "spicy"], 0.7),
            _draft("fall", ["amber", "woody"], 0.9),
        ]
    )
    assert consolidated.target_season == "winter"
    assert "amber" in consolidated.acceptable_accords
    assert "woody" in consolidated.acceptable_accords
    assert consolidated.confidence == 0.8
