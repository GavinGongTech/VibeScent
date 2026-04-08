# Karan

Role: Fragrance Representation Lead

Last updated: April 7, 2026

## Scope

You own:

- fragrance text generation
- structured fragrance attributes
- note-family and accord mapping

## Main Deliverables

### Week 2

- generate `retrieval_text`
- generate `display_text`
- compute structured fragrance attributes
- validate that similar fragrances cluster together

### Week 3

- refine representation quality based on benchmark failures
- support reranker inputs with cleaner structured fields

## Text Strategy

### `retrieval_text`

Rich, schema-controlled, optimized for ranking.

### `display_text`

More expressive, optimized for the demo.

Generation method:

- LLM-assisted templates

## Required Outputs

- fragrance text generation script or notebook
- final `retrieval_text`
- final `display_text`
- structured attribute table

## Interfaces You Depend On

From Darren:

- canonical fragrance table

From Harsh:

- text embedding pipeline

## Success Criteria

- fragrance text is rich without becoming noisy
- structured attributes improve scoring stability
- the representation supports both retrieval and polished UI copy
