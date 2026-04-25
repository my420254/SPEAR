
***

# SPEAR: Decoupled Biaffine Scoring with Span-Aware Representations and Consistency Regularisation for Emotion-Cause Pair Extraction

本仓库包含了论文 **"SPEAR: Decoupled Biaffine Scoring with Span-Aware Representations and Consistency Regularisation for Emotion-Cause Pair Extraction"** 的官方 PyTorch 实现代码。该论文已被定为向 *Information Processing & Management (IPM)* 期刊提交的候选稿件。

## 📌 项目概述 (Overview)

情感-原因对提取 (Emotion-Cause Pair Extraction, ECPE) 旨在从文档中联合提取情感子句及其对应的原因子句。为解决现有方法中存在的表示耦合问题与稀疏数据带来的过拟合问题，我们提出了 **SPEAR** 模型：
- **S**pan-Aware Representations (跨度感知表示)：通过聚合子句内的 Token 级信息来增强基于 CLS 的粗粒度特征。
- **P**air **E**xtraction with **A**ttention and **R**egularisation：
  - 引入了基于低秩自适应 (DoRA) 优化的双仿射评分机制 (DoRA-Biaffine)，在降低参数量的同时实现了情感与原因的解耦打分。
  - 引入 R-Drop (Consistency Regularisation) 策略，通过强制两次前向传播的输出分布一致，以缓解模型在小样本上的过拟合。

本代码库包含主实验 (10折交叉验证)、完整消融实验、超参数敏感度分析 (阈值与窗口)、显著性检验 (Wilcoxon test) 以及高标准的论文图表生成流水线。

## 🗂 代码库结构 (Repository Structure)

```text
.
├── ecpe.py                 # 核心训练脚本（包含模型定义、数据加载、训练评估、消融实验与显著性检验）
├── create_image.py         # 高质量论文图表生成脚本（基于 Matplotlib）
├── Times_New_Roman.ttf     # 绘图字体文件 (保证期刊图表格式规范)
├── README.md               # 本文档
└── data/                   # 存放 ECPE 数据集 (请确保划分为 fold1~fold10 的 train/test.json)
```

## ⚙️ 环境依赖 (Dependencies)

在运行代码前，请确保安装以下依赖。推荐使用 Python 3.8+ 及 PyTorch 1.12+ 版本的环境。

```bash
pip install torch numpy scipy scikit-learn transformers matplotlib
```

**预训练模型准备：**
代码默认使用 `hfl/chinese-roberta-wwm-ext` 模型。请将其下载至本地或设置合适的 HuggingFace 缓存路径，并在 `ecpe.py` 中正确配置 `MODEL_PATH`。

## 🚀 运行实验 (Running Experiments)

### 1. 模型训练与评估 (Training & Evaluation)
主控脚本 `ecpe.py` 已内置了主模型及各个变体（消融实验）的执行队列，并采用绝对安全的随机数隔离与断点续跑设计。

```bash
# 默认使用 GPU 1 运行所有实验，结果输出至 ./paper_run 目录
python ecpe.py --gpu 1 --out_dir ./paper_run
```

**执行流程说明：**
1. **主实验 (MAIN_Full)**: 运行完整的 SPEAR 模型，执行 10 折交叉验证。
2. **消融实验 (Ablations)**: 依次运行去掉 DoRA、SpanRepr、RDrop 以及 Biaffine 模块的基础版本对比实验 (`ABL_PureBase`, `ABL_woDoRA`, `ABL_woBiaffine` 等)。
3. **超参数实验 (Hyperparameters)**: 自动遍历不同的 DoRA 秩 ($r \in \{4, 8, 16, 32, 64\}$) 和解码窗口 ($W \in \{1, 2, 3, 4, 5, \infty\}$)。
4. **日志与持久化**: 所有的 Logits、模型预测结果、错误案例收集及显著性检验 (Wilcoxon Test) 将自动汇总至 `LEADERBOARD.csv` 和 `paper_run/figures/` 下的 JSON/CSV 文件中。

### 2. 论文图表生成 (Figure Generation)
实验运行完毕且数据均落盘于 `paper_run/` 后，可执行绘图脚本生成符合 IPM 期刊规格的高清图表 (PDF格式，300 DPI)。

```bash
python create_image.py
```

该脚本将读取 `paper_run/figures/` 下的统计数据，并在 `figures/` 目录下生成以下核心图表：
- `boxplot_folds.pdf`: 各模型在 10 折交叉验证中的 F1 分数分布箱线图。
- `ablation_cumulative.pdf`: 棒棒糖图 (Lollipop chart)，展示各组件累加与移除的消融效果。
- `rank_ablation.pdf` / `window_sensitivity_test.pdf`: 超参数折线图，用于评估 DoRA Rank $r$ 与解码窗口 $W$ 的鲁棒性。
- `significance_dotplot.pdf`: 模型改进的置信区间 (95% CI) 与显著性检验点图。
- `tsne_visualization.pdf`: 情感与原因特征解耦的 T-SNE 可视化 (仅需主实验落盘 `.npz` 数据)。

## 📊 主要实验结果 (Main Results)

在基准 ECPE 数据集 (10折交叉验证) 上，SPEAR 表现出了优越的性能：

| 模型变体 | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: |
| RoBERTa + Biaffine (Base) | - | - | 基线指标 |
| **SPEAR (Ours)** | **~** | **~** | **~77.22** |

*(详细指标请运行代码后参阅自动生成的 `table1_main.json` 及 `table2_ablation.csv`)*

**统计显著性测试：**
脚本内置非参数 Wilcoxon 符号秩检验。运行完毕后会在 `figures/significance.json` 中明确列出模型相较于基线（如纯 Biaffine 或 w/o RDrop）的性能提升是否满足 $p < 0.05$。


