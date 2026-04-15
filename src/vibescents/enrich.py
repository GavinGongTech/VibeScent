"""Enrich fragrance rows with structured vibe attributes and retrieval text."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import ValidationError

from vibescents.schemas import EnrichmentSchemaV2
from vibescents.settings import Settings

GEMINI_ENRICHMENT_MODEL = "gemini-3-flash-preview"
QWEN_ENRICHMENT_MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
DEFAULT_BATCH_SIZE = 16
DELAY_BETWEEN_BATCHES = 1.0

ENRICHMENT_COLUMNS = list(EnrichmentSchemaV2.model_fields.keys())

SYSTEM_PROMPT = """\
You are a fragrance expert. Given a perfume's metadata, generate structured vibe attributes.

Rules:
- formality: 0.0 = very casual, 1.0 = black tie formal
- fresh_warm: 0.0 = crisp/fresh, 1.0 = warm/cozy
- day_night: 0.0 = daytime, 1.0 = evening/night
- character_tags: 3-5 short adjectives
- vibe_sentence: one sentence grounded in the given metadata
- likely_occasion: one primary occasion label
- likely_season: one of spring/summer/fall/winter/all-season
- longevity: concise duration label (e.g., short/moderate/long)
- projection: concise projection label (e.g., intimate/moderate/strong)
- mood_tags: at least one mood-oriented tag
- color_palette: at least one color descriptor
"""


class EnrichmentClient(Protocol):
    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        """Generate one enrichment object from a prompt."""


@dataclass
class GeminiEnrichmentClient:
    model_name: str = GEMINI_ENRICHMENT_MODEL
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        from google.genai import types

        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=EnrichmentSchemaV2,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, EnrichmentSchemaV2):
                    return parsed
                if parsed is not None:
                    return EnrichmentSchemaV2.model_validate(parsed)
                return EnrichmentSchemaV2.model_validate_json(response.text)
            except Exception as exc:
                last_error = exc
                retryable = "429" in str(exc) or "503" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if retryable and attempt < 4:
                    delay = min(60.0, 2.0 ** (attempt + 1))
                    time.sleep(delay)
                    continue
                raise
        assert last_error is not None
        raise last_error


@dataclass
class QwenOutlinesEnrichmentClient:
    model_name: str = QWEN_ENRICHMENT_MODEL

    def __post_init__(self) -> None:
        self._generator = _build_outlines_generator(self.model_name)

    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        raw = self._generator(f"{SYSTEM_PROMPT}\n\n{prompt}")
        parsed = _parse_enrichment(raw)
        if parsed is not None:
            return parsed
        repaired = _repair_payload(raw)
        parsed = _parse_enrichment(repaired)
        if parsed is not None:
            return parsed
        raise ValueError("Outlines output could not be parsed into EnrichmentSchemaV2.")


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
    return outlines.generate.json(model, EnrichmentSchemaV2)


def _repair_payload(payload: Any) -> Any:
    try:
        from json_repair import repair_json
    except ImportError:
        return payload
    return repair_json(str(payload))


def _parse_enrichment(payload: Any) -> EnrichmentSchemaV2 | None:
    try:
        if isinstance(payload, EnrichmentSchemaV2):
            return payload
        if isinstance(payload, dict):
            return EnrichmentSchemaV2.model_validate(payload)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return EnrichmentSchemaV2.model_validate(parsed)
        return None
    except (ValidationError, TypeError, ValueError):
        return None


def _build_prompt(row: pd.Series) -> str:
    parts = [f"Name: {row.get('name', '')}"]
    for column, label in [
        ("brand", "Brand"),
        ("top_notes", "Top notes"),
        ("middle_notes", "Middle notes"),
        ("base_notes", "Base notes"),
        ("main_accords", "Accords"),
        ("gender", "Gender"),
        ("concentration", "Concentration"),
        ("category", "Category"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def _shrink_prompt(prompt: str, factor: float = 0.7) -> str:
    clipped = prompt[: max(1, int(len(prompt) * factor))]
    return clipped if clipped.endswith("\n") else f"{clipped}\n"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _append_failure_record(
    failures_path: str | None,
    *,
    row_index: int,
    prompt: str,
    error: str,
) -> None:
    if failures_path is None:
        return
    record = {
        "row_index": row_index,
        "error": error,
        "prompt": prompt,
    }
    with open(failures_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _resolve_client(
    *,
    provider: str,
    model_name: str | None,
) -> EnrichmentClient:
    if provider == "gemini":
        return GeminiEnrichmentClient(model_name=model_name or GEMINI_ENRICHMENT_MODEL)
    if provider == "qwen":
        return QwenOutlinesEnrichmentClient(model_name=model_name or QWEN_ENRICHMENT_MODEL)
    raise ValueError(f"Unsupported provider: {provider}")


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    provider: str = "qwen",
    model_name: str | None = None,
    max_rows: int | None = None,
    resume_from: int = 0,
    checkpoint_path: str | None = None,
    failures_path: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Enrich dataframe rows with EnrichmentSchemaV2 fields."""
    client = _resolve_client(provider=provider, model_name=model_name)

    work = df.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in work.columns:
            work[column] = pd.Series([None] * len(work), dtype="object")

    end = min(len(work), resume_from + max_rows) if max_rows else len(work)
    subset = work.iloc[resume_from:end]

    total = len(subset)
    processed = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = subset.iloc[batch_start:batch_end]

        for idx, row in batch.iterrows():
            prompt = _build_prompt(row)
            try:
                record = client.generate(prompt)
            except Exception as exc:
                try:
                    record = client.generate(_shrink_prompt(prompt))
                except Exception as retry_exc:
                    failed += 1
                    _append_failure_record(
                        failures_path,
                        row_index=int(idx),
                        prompt=prompt,
                        error=f"{exc}; retry={retry_exc}",
                    )
                    continue

            for column in ENRICHMENT_COLUMNS:
                work.at[idx, column] = _serialize_value(getattr(record, column))
            processed += 1

        done = min(batch_end, total)
        print(f"  [{done}/{total}] processed={processed} failed={failed}")
        if checkpoint_path:
            work.to_csv(checkpoint_path, index=False)
        if batch_end < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\nEnrichment complete: {processed} ok, {failed} failed out of {total}")
    return work


def _build_retrieval_text(row: pd.Series) -> str:
    parts: list[str] = []

    name = row.get("name", "")
    brand = row.get("brand", "")
    if pd.notna(brand) and brand:
        parts.append(f"Brand: {brand} | Name: {name}")
    else:
        parts.append(f"Name: {name}")

    if pd.notna(row.get("main_accords")):
        parts.append(f"Accords: {row['main_accords']}")

    note_parts = []
    for column, label in [("top_notes", "Top"), ("middle_notes", "Heart"), ("base_notes", "Base")]:
        value = row.get(column)
        if pd.notna(value):
            note_parts.append(f"{label}: {value}")
    if note_parts:
        parts.append(" | ".join(note_parts))

    if pd.notna(row.get("likely_season")):
        parts.append(f"Season: {row['likely_season']}")
    if pd.notna(row.get("likely_occasion")):
        parts.append(f"Best for: {row['likely_occasion']}")
    if pd.notna(row.get("formality")):
        formality = float(row["formality"])
        level = "low" if formality < 0.33 else "medium" if formality < 0.67 else "high"
        parts.append(f"Formality: {level}")

    for field_name, label in [
        ("character_tags", "Character"),
        ("mood_tags", "Mood"),
        ("color_palette", "Palette"),
    ]:
        value = row.get(field_name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, str) and value:
            parts.append(f"{label}: {value}")
        elif isinstance(value, list) and value:
            parts.append(f"{label}: {', '.join(value)}")

    if pd.notna(row.get("longevity")):
        parts.append(f"Longevity: {row['longevity']}")
    if pd.notna(row.get("projection")):
        parts.append(f"Projection: {row['projection']}")
    if pd.notna(row.get("vibe_sentence")):
        parts.append(f"Vibe: {row['vibe_sentence']}")

    return " | ".join(parts)


def build_retrieval_text(df: pd.DataFrame) -> pd.DataFrame:
    """Build retrieval_text from raw and enriched fields."""
    work = df.copy()

    def _safe_parse_json_array(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    saved_columns: dict[str, pd.Series] = {}
    for column in ("character_tags", "mood_tags", "color_palette"):
        if column in work.columns:
            saved_columns[column] = work[column].copy()
            work[column] = work[column].apply(_safe_parse_json_array)

    work["retrieval_text"] = work.apply(_build_retrieval_text, axis=1)
    for column, saved in saved_columns.items():
        work[column] = saved
    return work


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Enrich fragrance dataset with vibe attributes")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--provider", choices=["qwen", "gemini"], default="qwen")
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--failures-log", default=None, help="Optional JSONL path for failed rows.")
    args = parser.parse_args()

    checkpoint_path = args.output_csv + ".ckpt"
    failures_path = args.failures_log or args.output_csv + ".failures.jsonl"

    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    if args.resume_from > 0 and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...")
        df_ckpt = pd.read_csv(checkpoint_path)
        for column in df_ckpt.columns:
            if column not in df.columns:
                df[column] = df_ckpt[column]
        df.update(df_ckpt)

    if "fragrance_id" not in df.columns:
        df.insert(0, "fragrance_id", df.index.astype(str))
        print(f"Added fragrance_id column (0 to {len(df)-1})")

    print(
        "Enriching rows "
        f"with provider={args.provider} model={args.model or 'default'} "
        f"starting at index {args.resume_from}..."
    )
    enriched = enrich_dataframe(
        df,
        provider=args.provider,
        model_name=args.model,
        max_rows=args.max_rows,
        resume_from=args.resume_from,
        checkpoint_path=checkpoint_path,
        failures_path=failures_path,
        batch_size=args.batch_size,
    )
    enriched = build_retrieval_text(enriched)
    enriched.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv} ({enriched.shape[0]} rows).")


if __name__ == "__main__":
    main()
