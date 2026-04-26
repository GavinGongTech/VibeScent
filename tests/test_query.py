from __future__ import annotations

from vibescents.query import context_to_query_string
from vibescents.schemas import ContextInput


def test_empty_context_returns_default() -> None:
    result = context_to_query_string(ContextInput())
    assert result == "elegant versatile fragrance suitable for any occasion"


def test_event_type_known() -> None:
    result = context_to_query_string(ContextInput(eventType="Gala"))
    assert "black tie" in result
    assert "|" not in result


def test_time_of_day_known() -> None:
    result = context_to_query_string(ContextInput(timeOfDay="Evening"))
    assert "evening" in result


def test_mood_known() -> None:
    result = context_to_query_string(ContextInput(mood="Mysterious"))
    assert "mysterious" in result


def test_all_known_fields_join_with_pipe() -> None:
    ctx = ContextInput(eventType="Date Night", timeOfDay="Evening", mood="Warm")
    result = context_to_query_string(ctx)
    parts = result.split(" | ")
    assert len(parts) == 3


def test_unknown_event_type_passes_through() -> None:
    result = context_to_query_string(ContextInput(eventType="Birthday Party"))
    assert "Birthday Party" in result


def test_unknown_time_passes_through() -> None:
    result = context_to_query_string(ContextInput(timeOfDay="Dawn"))
    assert "Dawn" in result


def test_custom_notes_appended() -> None:
    result = context_to_query_string(ContextInput(customNotes="smells like rain"))
    assert "smells like rain" in result


def test_custom_notes_stripped_of_whitespace() -> None:
    result = context_to_query_string(ContextInput(customNotes="  notes  "))
    assert result.endswith("notes")


def test_event_plus_custom_notes() -> None:
    ctx = ContextInput(eventType="Gala", customNotes="extra note")
    result = context_to_query_string(ctx)
    assert " | " in result
    assert "extra note" in result
