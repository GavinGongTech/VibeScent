# Darren

Role: Fragrance Data Lead

Last updated: April 7, 2026

## Scope

You own:

- fragrance dataset selection
- schema definition
- data loading
- normalization and missing-value handling

## Main Deliverables

### Week 2

- choose one primary fragrance dataset
- document the schema
- clean core fields
- export the canonical fragrance table

### Week 3

- improve coverage only if it does not break the canonical schema
- support targeted enrichment where the benchmark exposes gaps

## Required Fields

- `fragrance_id`
- `brand`
- `name`
- `notes`
- `accords`
- any available season, occasion, and gender metadata

## Required Outputs

- source dataset decision note
- canonical fragrance table
- data quality summary
- missingness summary

## Constraints

- do not merge multiple messy datasets before one clean table exists
- do not let schema drift across branches

## Success Criteria

- the team has one stable fragrance table
- downstream branches can consume it without manual cleanup
