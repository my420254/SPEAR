# SPEAR: Decoupled Biaffine Scoring with Span-Aware Representations and Consistency Regularisation for Emotion-Cause Pair Extraction

This repository contains the official PyTorch implementation of **SPEAR**, a lightweight ECPE framework.

## What this work solves

Emotion-Cause Pair Extraction is a structural extraction task where precision matters more than parameter count.

The main failure modes are:

- over-coupled clause representations
- weak boundary awareness
- overconfident false positives on sparse data

## Core idea

SPEAR introduces four components:

- **Span-Aware Clause Encoder**: recovers token-level implicit cues inside each clause
- **DoRA-based Biaffine Scorer**: separates emotion and cause scoring with a more stable low-rank parameterization
- **R-Drop Regularisation**: penalizes inconsistent predictions across two forward passes
- **Window-Constrained Decoding**: filters long-range noise with a locality prior

## My contribution

- Designed the model and evaluation pipeline
- Completed ablation, sensitivity, and significance testing
- Generated the paper figures and supporting analysis

## Main result

Under strict 10-fold cross validation on the Chinese ECPE benchmark, SPEAR reaches **77.24% F1**, improving the previous best baseline by **0.87 points**.

## Repository contents

- `ecpe.py`: training / evaluation / ablation / significance test runner
- `create_image.py`: figure generation script
- `data/`: ECPE folds

## Status

KBS submission version.

