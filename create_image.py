"""
create_image.py  —  DoRA-SpanEC ECPE Paper: High-End Journal Figures
===================================================================================
Run:  python create_image.py
Output: figures/ directory (PDF + PNG, 300 dpi)
"""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines # 🌟 新增用于生成隐形图例
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

PAPER_RUN_DIR = "./paper_run"
FIG_DATA_DIR = os.path.join(PAPER_RUN_DIR, "figures")
TSNE_DATA_DIR = os.path.join(PAPER_RUN_DIR, "tsne")
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ── Font setup ──
FONT_PATH = "imes_New_Roman.ttf"
if os.path.exists(FONT_PATH):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_PATH)
    FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_NAME = "DejaVu Serif"

plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : [FONT_NAME, "Times New Roman", "DejaVu Serif"],
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 12,
    "xtick.labelsize"   : 10.5,
    "ytick.labelsize"   : 10.5,
    "legend.fontsize"   : 10.5,
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth"   : 2.0,
    "lines.markersize"  : 7,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.linestyle"    : "--",
    "grid.linewidth"    : 0.5,
    "grid.alpha"        : 0.5,
})

# ── Color Palette (Journal Style) ──
C = {
    "blue"   : "#1f77b4",
    "red"    : "#d73027",
    "green"  : "#4daf4a",
    "orange" : "#f46d43",
    "purple" : "#762a83",
    "gray"   : "#969696",
    "light_b": "#abd9e9"
}

def load_json(filename):
    path = os.path.join(FIG_DATA_DIR, filename)
    if not os.path.exists(path):
        path = os.path.join(PAPER_RUN_DIR, filename)
        if not os.path.exists(path):
            print(f"[Warning] Data file missing: {filename}")
            return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_boxplot():
    data = load_json("boxplot.json")
    if not data: return
    
    # 🌟 修改点 1：将 "Full\n(Ours)" 更改为品牌感更强的 "SPEAR\n(Ours)"
    models = ["Base", "w/o DoRA", "w/o Span", "w/o Biaffine", "SPEAR\n(Ours)"]
    keys = ["ABL_PureBase", "ABL_woDoRA", "ABL_woSpan", "ABL_woBiaffine", "MAIN_Full"]
    palette = [C["gray"], C["orange"], C["green"], C["purple"], C["blue"]]

    data_pct = [data.get(k, {}).get("fold_f1s", [0]*10) for k in keys]

    # 画布高度稍微缩短一点，因为去掉了底部的图例
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bp = ax.boxplot(data_pct, patch_artist=True, widths=0.5,
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="none", markeredgecolor="gray"),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color="gray"),
                    capprops=dict(linewidth=1.5, color="gray"))

    for patch, fc in zip(bp["boxes"], palette):
        patch.set_facecolor(fc)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for i, d in enumerate(data_pct, 1):
        ax.scatter(i, np.mean(d), marker="x", s=50, color="black", zorder=6, lw=1.5)

    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels(models)
    ax.set_ylabel(r"F$_1$ Score (%)")
    
    all_vals = [v for sub in data_pct for v in sub]
    ax.set_ylim(np.floor(min(all_vals))-1, np.ceil(max(all_vals))+1)

    # ★ 彻底删除了这里的 ax.legend(...) 代码，告别多余的图例

    # 🌟 修改点 2：注释掉自带标题，交给 LaTeX 的 caption
    # ax.set_title(r"Per-Fold F$_1$ Distribution")
    fig.savefig(f"{OUT}/boxplot_folds.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation_lollipop():
    data = load_json("table2_ablation.json")
    if not data: return
    
    labels = [item["model"] for item in data]
    F1s    = [item["F1"] for item in data]
    xs = np.arange(len(labels))

    palette = [C["gray"], "#fcae91", "#f46d43", C["blue"], 
               "#abd9e9", "#74add1", "#4575b4", "#e0f3f8"][:len(labels)]

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    y_min = np.floor(min(F1s)) - 1.0

    for i, (v, col) in enumerate(zip(F1s, palette)):
        ax.plot([i, i], [y_min, v], color=col, lw=3.0, alpha=0.8, zorder=2)
        ax.scatter(i, v, color=col, s=130, zorder=3, marker="o", edgecolor="white", lw=1.5)
        weight = "bold" if "Full" in labels[i] else "normal"
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight=weight)

    full_f1 = next((item["F1"] for item in data if "Full" in item["model"]), None)
    if full_f1:
        ax.axhline(full_f1, color=C["blue"], linestyle="--", lw=1.2, alpha=0.5, zorder=1)

    ax.axvline(3.5, color="gray", linestyle=":", lw=1.5, zorder=1)
    ax.text(1.5, np.ceil(max(F1s)) + 0.8, "Cumulative Addition", ha="center", fontsize=11.5, fontstyle="italic", color="#555555")
    ax.text(5.5, np.ceil(max(F1s)) + 0.8, "Component Removal", ha="center", fontsize=11.5, fontstyle="italic", color="#555555")

    clean_labels = []
    for l in labels:
        if "Base" in l or "RoBERTa" in l: clean_labels.append("Base")
        elif "+ RDrop" == l or "RDropOnly" in l: clean_labels.append("+ RDrop")
        elif "+ C1 DoRA-Biaffine" in l: clean_labels.append("+ C1\nDoRA-Biaffine")
        elif "Full" in l and "w/o" not in l: clean_labels.append("Full\nModel")
        elif "w/o C1 DoRA" in l: clean_labels.append("w/o\nC1 DoRA")
        elif "w/o C2 SpanRepr" in l: clean_labels.append("w/o\nC2 SpanRepr")
        elif "w/o C3 Biaffine" in l: clean_labels.append("w/o\nC3 Biaffine")
        elif "w/o RDrop" in l: clean_labels.append("w/o\nRDrop")
        else: clean_labels.append(l)

    ax.set_xticks(xs)
    ax.set_xticklabels(clean_labels, rotation=0, fontsize=10.5)
    ax.set_ylabel(r"Avg. F$_1$ (%)")
    
    ax.set_ylim(y_min, np.ceil(max(F1s)) + 1.5)
    
    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title("Ablation Study Results", pad=15)

    ax.spines['bottom'].set_position(('data', y_min))

    fig.savefig(f"{OUT}/ablation_cumulative.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/ablation_cumulative.png")
    plt.close(fig)


def fig_rank_ablation():
    data = load_json("rank_ablation.json")
    if not data: return
    
    ranks = sorted([int(k) for k in data.keys()])
    F1s   = [data[str(r)]["avg_F1"]*100 for r in ranks]
    xs = np.arange(len(ranks))
    
    # 🌟 修改点 3: 强制与 window 图保持一样的尺寸 (6.0, 4.5)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(xs, F1s, marker="o", markersize=8, color=C["blue"], lw=2.5, mfc="white", mec=C["blue"], mew=2)
    
    for i, v in enumerate(F1s):
        weight = "bold" if v==max(F1s) else "normal"
        ax.text(xs[i], v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight=weight)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("DoRA Rank $r$")
    ax.set_ylabel(r"Avg. F$_1$ (%)")
    ax.set_ylim(np.floor(min(F1s))-0.5, np.ceil(max(F1s))+0.8)
    
    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title("Effect of DoRA Rank on Performance")

    # 🌟 修改点 3: 加入隐形的图例占位。保证本图 bbox_inches="tight" 裁剪出的下边距与 window 图严丝合缝
    dummy_line = mlines.Line2D([], [], color='none', label=' ')
    ax.legend(handles=[dummy_line], loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)

    fig.savefig(f"{OUT}/rank_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_window_sensitivity():
    data = load_json("window_testset.json")
    if not data: return
    
    wins = sorted([int(k) for k in data.keys()])
    xlabels = [str(w) if w < 900 else r"$\infty$" for w in wins]
    xs = np.arange(len(wins))

    P_v = [data[str(w)]["avg_P"]*100 for w in wins]
    R_v = [data[str(w)]["avg_R"]*100 for w in wins]
    F_v = [data[str(w)]["avg_F1"]*100 for w in wins]

    # 🌟 修改点 3: 强制与 DoRA rank 图保持一样的尺寸 (6.0, 4.5)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    ax.plot(xs, F_v, marker="o", color=C["blue"],  label=r"F$_1$", lw=2.5, zorder=4)
    ax.plot(xs, P_v, marker="s", color=C["red"],   label="Precision", lw=1.5, linestyle="--", alpha=0.8)
    ax.plot(xs, R_v, marker="^", color=C["green"], label="Recall",    lw=1.5, linestyle="--", alpha=0.8)

    opt_idx = 2
    if wins[opt_idx] == 3:
        ax.axvspan(xs[opt_idx]-0.4, xs[opt_idx]+0.4, color=C["gray"], alpha=0.15, zorder=0, lw=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Decoding Window Size $W$")
    ax.set_ylabel("Metric Value (%)")
    
    min_val = min(min(P_v), min(R_v), min(F_v))
    max_val = max(max(P_v), max(R_v), max(F_v))
    ax.set_ylim(np.floor(min_val)-1, np.ceil(max_val)+1)
    
    # 图例位置保持不变，上一个函数里的隐形图例就是和这里对标的
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
    
    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title("Effect of Decoding Window (Test Set)")

    fig.savefig(f"{OUT}/window_sensitivity_test.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_threshold_sensitivity():
    data = load_json("sens_threshold.json")
    if not data: return
    
    thrs = sorted([float(k) for k in data.keys()])
    xlabels = [f"{t:.1f}" for t in thrs]
    xs = np.arange(len(thrs))

    P_v = [data[f"{t:.2f}"]["avg_P"]*100 for t in thrs]
    R_v = [data[f"{t:.2f}"]["avg_R"]*100 for t in thrs]
    F_v = [data[f"{t:.2f}"]["avg_F1"]*100 for t in thrs]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(xs, F_v, marker="o", color=C["blue"],  label=r"F$_1$", lw=2.5, zorder=4)
    ax.plot(xs, P_v, marker="s", color=C["red"],   label="Precision", lw=1.5, linestyle="--", alpha=0.8)
    ax.plot(xs, R_v, marker="^", color=C["green"], label="Recall",    lw=1.5, linestyle="--", alpha=0.8)

    opt_idx = thrs.index(0.0) if 0.0 in thrs else None
    if opt_idx is not None:
        ax.axvspan(xs[opt_idx]-0.4, xs[opt_idx]+0.4, color=C["gray"], alpha=0.15, zorder=0, lw=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(r"Decoding Threshold $\tau$")
    ax.set_ylabel("Metric Value (%)")
    
    min_val = min(min(P_v), min(R_v), min(F_v))
    max_val = max(max(P_v), max(R_v), max(F_v))
    ax.set_ylim(np.floor(min_val)-1, np.ceil(max_val)+1)
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
    
    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title("Effect of Decoding Threshold (Train Set)")

    fig.savefig(f"{OUT}/threshold_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_significance_heatmap():
    data = load_json("significance.json")
    if not data: return
    
    order_keys = ["MAIN_vs_ABL_PureBase", "MAIN_vs_ABL_RDropOnly", "MAIN_vs_ABL_DoRA_RDrop",
                  "MAIN_vs_ABL_woDoRA", "MAIN_vs_ABL_woSpan", "MAIN_vs_ABL_woBiaffine", "MAIN_vs_ABL_woRDrop"]
    
    models, deltas, ci_lo, ci_hi, is_sig = [], [], [], [], []
    for k in order_keys:
        if k in data:
            d = data[k]
            m_name = k.replace("MAIN_vs_ABL_", "").replace("MAIN_vs_", "").replace("Pure", "")
            if "wo" in m_name:
                m_name = m_name.replace("wo", "w/o\n")
            else:
                m_name = m_name.replace("_", "\n+")
                if m_name != "Base": m_name = "+" + m_name
            models.append(m_name)
            deltas.append(d["mean_diff"] * 100)
            ci_lo.append(d["boot_CI_95"][0] * 100)
            ci_hi.append(d["boot_CI_95"][1] * 100)
            is_sig.append(d["significant_p05"])

    xs = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    colors_dot = [C["blue"] if sig else C["gray"] for sig in is_sig]
    
    for i, (d, lo, hi, col) in enumerate(zip(deltas, ci_lo, ci_hi, colors_dot)):
        ax.plot([i,i], [lo,hi], color=col, lw=3.0, alpha=0.7, zorder=2)
        ax.scatter(i, d, color=col, s=80, zorder=3, marker="o", edgecolor="white")

    ax.axhline(0, color="black", lw=1.2, linestyle="--", alpha=0.5)

    for i, (sig, hi) in enumerate(zip(is_sig, ci_hi)):
        y_pos = hi + 0.15
        if sig:
            ax.text(i, y_pos, "*", ha="center", va="bottom", fontsize=14, color=C["blue"])
        else:
            ax.text(i, y_pos, "n.s.", ha="center", va="bottom", fontsize=10, color=C["gray"])

    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=0, fontsize=10.5) 
    ax.set_ylabel(r"$\Delta$F$_1$ (%) vs. Variant")
    
    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title(r"Mean Improvement of Full Model (95% CI)")
    
    ax.set_ylim(np.floor(min(ci_lo))-0.5, np.ceil(max(ci_hi))+0.8)

    sig_patch   = mpatches.Patch(color=C["blue"], alpha=0.7, label="$p < 0.05$ (Significant)")
    nosig_patch = mpatches.Patch(color=C["gray"], alpha=0.7, label="$p \geq 0.05$ (n.s.)")
    ax.legend(handles=[sig_patch, nosig_patch], loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=2)

    fig.savefig(f"{OUT}/significance_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tsne():
    npz_files = [f for f in os.listdir(TSNE_DATA_DIR) if f.endswith(".npz")]
    if not npz_files: return
    
    file_path = os.path.join(TSNE_DATA_DIR, npz_files[0])
    data = np.load(file_path)
    h_emo, h_cause, labels = data['h_emo'], data['h_cause'], data['labels']

    N_MAX = 1500
    if len(h_emo) > N_MAX:
        idx = np.random.choice(len(h_emo), N_MAX, replace=False)
        h_emo, h_cause, labels = h_emo[idx], h_cause[idx], labels[idx]

    print("[TSNE] Running dimensionality reduction...")
    combined = np.vstack([h_emo, h_cause])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embed = tsne.fit_transform(combined)
    
    emb_e = embed[:len(h_emo)]
    emb_c = embed[len(h_emo):]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    
    neg = (labels == 0)
    ax.scatter(emb_e[neg, 0], emb_e[neg, 1], c="#cccccc", alpha=0.4, s=20, label="Emotion (Neg)")
    ax.scatter(emb_c[neg, 0], emb_c[neg, 1], c="#e0e0e0", alpha=0.4, s=20, label="Cause (Neg)")
    
    pos = (labels == 1)
    ax.scatter(emb_e[pos, 0], emb_e[pos, 1], c=C["red"], alpha=0.9, s=50, marker="^", edgecolor="white", lw=0.5, label="Emotion (Pos)")
    ax.scatter(emb_c[pos, 0], emb_c[pos, 1], c=C["blue"], alpha=0.9, s=50, marker="o", edgecolor="white", lw=0.5, label="Cause (Pos)")

    # 🌟 修改点 2：注释掉自带标题
    # ax.set_title("T-SNE Embeddings of Decoupled Representations")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=2)

    fig.savefig(f"{OUT}/tsne_visualization.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating High-End Journal Figures")
    print("=" * 60)

    fig_boxplot()
    fig_ablation_lollipop()
    fig_rank_ablation()
    fig_window_sensitivity()
    fig_threshold_sensitivity()
    fig_significance_heatmap()
    
    if os.path.exists(TSNE_DATA_DIR):
        fig_tsne()

    print("=" * 60)
    print(f"  Done! All outputs in ./{OUT}/")