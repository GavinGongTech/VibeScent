# VibeScent Evaluation Plan

Last updated: April 7, 2026

## Goal

Measure whether the system is actually improving recommendation quality without pretending that AI-written labels are the same thing as human ground truth.

## Benchmark Structure

The benchmark contains 20 end-to-end cases.

Each case includes:

- one outfit image
- one occasion description
- target style attributes
- acceptable fragrance neighborhoods
- disallowed traits

## Label Generation

The benchmark is AI-assisted.

Primary label generator:

- Gemini 3.1 Pro Preview

Why this model:

- it supports Batch API
- it supports structured outputs
- it accepts multimodal inputs, which matters for outfit-image benchmark cases

Generation method:

- batch or looped structured-output requests
- three independent generations per case
- keep only cases with strong agreement across runs

Required output schema per case:

- `case_id`
- `occasion_text`
- `target_formality`
- `target_season`
- `target_day_night`
- `target_fresh_warm`
- `acceptable_accords`
- `acceptable_note_families`
- `disallowed_traits`
- `example_good_fragrances`
- `confidence`

## Primary Scoring

Do not use the same model as both label writer and primary judge.

Primary score should be metadata-based.

For each retrieved fragrance, score:

- accord match
- note-family match
- season match
- formality match
- day or night match
- fresh versus warm match

From this, compute:

- `attribute_match@3`
- `attribute_match@5`
- `neighborhood_hit@3`
- `neighborhood_hit@5`

This keeps the main metric tied to fragrance metadata instead of pure model opinion.

## Secondary Scoring

Use an LLM judge only as a secondary evaluator.

Recommended judge:

- Gemini 2.5 Pro

Reason:

- it separates the judge from the label generator
- it reduces direct self-confirmation bias

Judge inputs:

- outfit image
- occasion text
- top candidates from the baseline
- top candidates from the reranker

Judge outputs:

- preferred shortlist
- short rationale
- confidence

## Required Reporting

Report all of the following:

- text-only retrieval performance
- image-only retrieval performance
- multimodal embedding retrieval performance
- late-fusion baseline performance
- reranked performance
- with and without `gemini-embedding-2`
- benchmark cases where reranking helps
- benchmark cases where reranking hurts

## Presentation Rule

When presenting results, call this what it is:

- an AI-assisted benchmark with metadata-based scoring

Do not call it:

- human-labeled ground truth

## Failure Conditions

The evaluation is not credible if:

- the same model generates labels and acts as the only judge
- the benchmark cases are not schema-consistent
- the system is only shown on hand-picked wins
- there is no baseline versus reranker comparison

## Official References

- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Batch API: https://ai.google.dev/gemini-api/docs/batch-api
- Gemini models: https://ai.google.dev/gemini-api/docs/models
