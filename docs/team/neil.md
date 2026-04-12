# Neil

Role: Image Retrieval Lead

Last updated: April 7, 2026

## Scope

You own:

- image-to-fragrance retrieval
- image preprocessing
- CLIP-only, CNN-only, and hybrid comparison

## Main Deliverables

### Week 2

- build the CLIP branch with OpenCLIP
- build the CNN branch with ResNet50
- build the hybrid branch with score fusion
- save image embeddings and score outputs
- produce similarity and nearest-neighbor artifacts

### Week 3

- integrate the winning image branch into the late-fusion baseline
- compare image-only versus fused retrieval
- compare CLIP/CNN/hybrid behavior against the shared-space `gemini-embedding-2` signal

## Model Choices

CLIP branch:

- OpenCLIP

CNN branch:

- ResNet50

Hybrid:

- score fusion first

Recommended initial rule:

`image_score = 0.70 * clip_score + 0.30 * cnn_score`

## Required Outputs

- image embedding artifacts
- image-to-fragrance score table
- CLIP versus CNN versus hybrid comparison
- one short note on failure modes

## Interfaces You Depend On

From Darren:

- cleaned fragrance table

From Karan:

- `retrieval_text`
- structured fragrance attributes

From Gavin:

- benchmark case file

## Success Criteria

- visually similar outfits score similarly
- image-only retrieval produces plausible fragrance candidates
- the hybrid is clearly defined and benchmarked
- the image branch still contributes unique value even when the multimodal embedding signal is added
