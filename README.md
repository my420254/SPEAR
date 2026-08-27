# SPEAR: Decoupled Biaffine Scoring with Span-Aware Representations and Consistency Regularisation for Emotion-Cause Pair Extraction

## 中文项目介绍

SPEAR 是我围绕情绪原因对抽取（Emotion-Cause Pair Extraction, ECPE）完成的结构化 NLP 工作，当前按 **Knowledge-Based Systems（KBS）** 投稿材料组织。KBS 是知识工程、人工智能和智能系统方向的高影响力期刊，常见分区口径较高；最终状态以官方录用结果为准。

ECPE 的目标是从文档中同时识别情绪子句和对应原因子句，并正确配对。它比普通分类任务更难，因为模型不仅要判断“哪里有情绪、哪里有原因”，还要解决跨子句关系建模、稀疏正样本、远距离噪声和配对误判。

## 核心方法

- **Span-Aware Clause Encoder**：从 clause 内部 token 表示恢复细粒度边界和局部语义，避免只用粗粒度句向量导致线索损失。
- **DoRA-based Biaffine Scorer**：用低秩稳定化的 biaffine 打分拆分 emotion/cause pair 建模，提高关系打分的表达能力和训练稳定性。
- **R-Drop Consistency Regularisation**：通过两次 forward 的一致性约束，降低小数据结构化抽取中的过拟合和过度自信。
- **Window-Constrained Decoding**：利用情绪和原因通常局部相关的先验，过滤不合理长距离候选，提升 precision。

## 面试展示重点

- **任务含金量**：ECPE 是典型文档级结构化抽取任务，比句子分类更考验表示学习、关系建模和解码设计。
- **方法组合能力**：把 span 表示、biaffine scoring、DoRA 参数化、R-Drop 正则和先验解码结合成完整 pipeline。
- **实验表现**：在中文 ECPE benchmark 严格 10-fold cross validation 下达到 **77.24% F1**，相对前序强 baseline 提升 **0.87**。
- **工程完整性**：包含主训练评测脚本、图表生成脚本、fold 数据、消融和显著性检验支持。
- **可讲难点**：正负样本极不平衡、pair 数量随 clause 数平方增长、跨句关系容易误配、小数据过拟合、precision 和 recall 的取舍。

## 技术关键词

`PyTorch` · `NLP` · `ECPE` · `Biaffine` · `DoRA` · `R-Drop` · `Document-level Extraction` · `Structured Prediction`

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
