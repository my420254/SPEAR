# SPEAR：面向情绪原因对抽取的 Span-Aware 关系建模框架

SPEAR 是一套面向情绪原因对抽取（Emotion-Cause Pair Extraction, ECPE）的结构化 NLP 框架，当前按 **Knowledge-Based Systems（KBS）** 投稿版本整理。项目聚焦文档级情绪子句与原因子句的联合识别和配对问题。

ECPE 的目标是从一段文档中同时找出情绪子句、原因子句，并判断两者是否构成有效 emotion-cause pair。它比普通分类任务更难，因为模型需要同时解决局部语义识别、跨子句关系建模、稀疏正样本学习和解码约束。

## 研究问题

ECPE 任务存在几个典型难点：

- pair 候选数量随 clause 数量平方增长，负样本远多于正样本；
- 情绪和原因可能跨句出现，关系边界容易混淆；
- 中文文档数据规模较小，模型容易过拟合；
- 只用粗粒度 clause 表示会丢失内部 token 线索；
- 高 recall 和高 precision 之间存在明显取舍。

SPEAR 的设计目标是在轻量模型规模下提升关系打分稳定性，并通过结构化解码减少不合理候选。

## 方法设计

### Span-Aware Clause Encoder

通过 clause 内部 token 表示恢复细粒度语义线索，避免只使用粗粒度句向量导致局部证据丢失。

### DoRA-Based Biaffine Scorer

使用 DoRA 思路稳定 biaffine 关系打分，将 emotion/cause pair 的关系建模从简单拼接分类提升为更强的双向交互打分。

### R-Drop Consistency Regularisation

通过两次 forward 的一致性约束降低小数据结构化抽取中的过拟合和过度自信，使模型在稀疏正样本场景下更稳定。

### Window-Constrained Decoding

引入情绪和原因通常局部相关的先验，过滤过远候选 pair，在降低误配的同时保持召回能力。

## 工程亮点

- `ecpe.py` 覆盖训练、推理、评测、消融、敏感性分析和显著性检验；
- 支持严格 10-fold cross validation；
- 保存 fold-level 指标、训练曲线、敏感性分析和 t-SNE 可视化数据；
- 代码中包含 `ECPE_Dataset`、`ECPE_Model`、`run_fold`、`run_sensitivity`、`significance_test` 等清晰模块；
- 可导出论文表格和图表，便于复现与审计。

## 结果摘要

在中文 ECPE benchmark 的严格 10-fold cross validation 设置下，SPEAR 达到 **77.24% F1**，相对前序强 baseline 提升 **0.87**。结果说明，span-aware 表示、biaffine scoring、DoRA 参数化和一致性正则的组合能够有效提升文档级 emotion-cause pair 抽取质量。

## 仓库结构

| 文件 | 说明 |
| --- | --- |
| `ecpe.py` | 主训练、评测、消融、敏感性分析和显著性检验脚本 |
| `create_image.py` | 图表生成脚本 |
| `data/` | ECPE 10-fold 数据 |
| `figures/` | 论文图表 |
| `paper_run/` | 实验输出与中间结果 |

## 运行方式

```bash
python ecpe.py --gpu 0 --out_dir ./paper_run
```

脚本会按配置执行 10-fold 训练评测，并输出 fold-level 指标、聚合结果和补充分析文件。

## 项目状态

- 论文状态：KBS 投稿版本；
- 代码状态：训练评测、消融、敏感性分析、显著性检验和图表生成已整理；
- 许可协议：MIT License。
