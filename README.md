# Outfit Vibe Classifier for Perfume Recommender System

This project implements an outfit vibe classifier using various deep learning approaches to recommend perfumes based on outfit vibes.

## Experiments

1. **CNN Baseline (ResNet-50)**: A convolutional neural network using ResNet-50 as the backbone for image classification.
2. **CNN + CLIP ViT Hybrid**: A hybrid model combining CNN features with CLIP Vision Transformer.
3. **CLIP ViT Standalone**: Using CLIP Vision Transformer directly for classification.

## Data

- **Source**: Fashionpedia
- **Labels**: Generated via CLIP zero-shot classification
  - Vibe (5 categories)
  - Formal (0.0-1.0 scale)
  - Season
  - Gender
  - Time of day
  - Frequency

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download Fashionpedia dataset and place in `data/fashionpedia/`

## Directory Structure

- `data/fashionpedia/`: Dataset storage and preprocessing
- `models/`: Model architecture definitions
- `experiments/`: Experiment-specific configurations and scripts
- `src/`: Shared utilities and data loaders
- `notebooks/`: Jupyter notebooks for data exploration and prototyping

## Usage

Run experiments using the scripts in `experiments/` directories.
