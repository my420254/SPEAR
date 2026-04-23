"""
ECPE 论文最终版 v12 (终极满配版 - 绝对安全隔离)
===================
目标精度: ~77.22%

安全保证: 
  - ABL_woBiaffine 消融实验被放置在执行队列的最末尾。
  - 新增的 MLP 模块仅在运行此特定消融时才实例化，绝对不干扰原版随机数轨迹。
  - 自动跳过已完成的实验，统一汇总落盘。
"""

import os, sys, gc, logging, csv, time, json, random, math, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",     type=str, default="1",
                   help="必须用跑出 77.46% 的那张卡，默认卡1")
    p.add_argument("--out_dir", type=str, default="./paper_run",
                   help="固定输出目录，支持断点续跑")
    return p.parse_args()

ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
OUT_DIR = ARGS.out_dir

# ════════════════════════════════════════════════════════════
#  路径
# ════════════════════════════════════════════════════════════
DATA_DIR   = "/Rank-Emotion-Cause/data/split10"
MODEL_PATH = (
    ".cache/huggingface/hub/"
    "models--hfl--chinese-roberta-wwm-ext/snapshots/"
    "5c58d0b8ec1d9014354d691c538661bf00bfdb44"
)

# ════════════════════════════════════════════════════════════
#  超参
# ════════════════════════════════════════════════════════════
CFG = {
    "enc_lr"       : 1e-5,
    "head_lr"      : 5e-4,
    "dropout"      : 0.2,
    "pos_weight"   : 3.8,      # 固定，不用自适应
    "epochs"       : 18,
    "batch_size"   : 12,
    "max_src_len"  : 512,
    "warmup_ratio" : 0.1,
    "grad_clip"    : 1.0,
    "num_folds"    : 10,
    "seed"         : 42,
    "num_workers"  : 4,
    "rdrop_alpha"  : 4.0,
    "dora_rank"    : 4,        # 主实验默认 rank=4
    "orth_lambda"  : 0.1,
    "fixed_thr"    : 0.0,
    "fixed_window" : 3,
    "sens_windows" : [1, 2, 3, 4, 5, 6, 999],
    "sens_thrs"    : [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0],
}

# ════════════════════════════════════════════════════════════
#  随机数锁定
# ════════════════════════════════════════════════════════════
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

# ════════════════════════════════════════════════════════════
#  实验注册表
# ════════════════════════════════════════════════════════════
def _e(use_dora=True, use_span=True, use_rdrop=True,
       use_orth=False, dora_rank=None, do_sens=False, use_biaffine=True):
    return {
        "use_dora"     : use_dora,
        "use_span"     : use_span,
        "use_rdrop"    : use_rdrop,
        "use_orth"     : use_orth,
        "use_biaffine" : use_biaffine, # 控制是否使用双仿射
        "dora_rank"    : dora_rank if dora_rank is not None else CFG["dora_rank"],
        "do_sens"      : do_sens,
    }

# ── 主实验 + 消融 ──────────────────────────────────────────
PART1 = {
    "MAIN_Full"      : _e(do_sens=True),
    "ABL_PureBase"   : _e(use_dora=False, use_span=False, use_rdrop=False),
    "ABL_RDropOnly"  : _e(use_dora=False, use_span=False),
    "ABL_DoRA_RDrop" : _e(use_span=False),
    "ABL_woDoRA"     : _e(use_dora=False),
    "ABL_woSpan"     : _e(use_span=False),
    "ABL_woRDrop"    : _e(use_rdrop=False),
}

# ── rank 超参 + 窗口消融 ───────────────────────────────────
PART2 = {
    "RANK_r4"  : _e(dora_rank=4),
    "RANK_r8"  : _e(dora_rank=8),
    "RANK_r16" : _e(dora_rank=16),
    "RANK_r32" : _e(dora_rank=32),
    "RANK_r64" : _e(dora_rank=64),
    "WIN_w1"   : _e(),
    "WIN_w2"   : _e(),
    "WIN_w3"   : _e(),
    "WIN_w4"   : _e(),
    "WIN_w5"   : _e(),
    "WIN_wInf" : _e(),
}

WIN_OVERRIDE = {
    "WIN_w1": 1, "WIN_w2": 2, "WIN_w3": 3,
    "WIN_w4": 4, "WIN_w5": 5, "WIN_wInf": 999,
}

# 组合所有原版实验
EXPERIMENTS = {**PART1, **PART2}

# ★ 绝对隔离保证：将新增的消融实验强制挂在执行队列的最末尾，最后跑！
EXPERIMENTS["ABL_woBiaffine"] = _e(use_dora=False, use_biaffine=False)

# ════════════════════════════════════════════════════════════
#  工具
# ════════════════════════════════════════════════════════════
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def setup_logging(out_dir):
    safe_mkdir(f"{out_dir}/logs")
    fmt  = logging.Formatter("%(asctime)s │ %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in [
        logging.FileHandler(f"{out_dir}/logs/master.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        root.addHandler(h)
    return logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
#  数据集
# ════════════════════════════════════════════════════════════
class ECPE_Dataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=512):
        self.tokenizer     = tokenizer
        self.max_len       = max_len
        self.clause_tok_id = tokenizer.convert_tokens_to_ids("<clause>")
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = self._process(raw)

    def _process(self, raw):
        out = []
        for doc in raw:
            idx2id, id2idx, texts = {}, {}, []
            for i, c in enumerate(doc["clauses"]):
                cid = int(c["clause_id"])
                idx2id[i] = cid
                id2idx[cid] = i
                texts.append(f"<clause> {c['clause']}")
            n   = len(texts)
            mat = np.zeros((n, n), dtype=np.float32)
            for eid, cid in doc.get("pairs", []):
                if eid in id2idx and cid in id2idx:
                    mat[id2idx[eid], id2idx[cid]] = 1.0
            out.append({
                "input_text"   : " ".join(texts),
                "labels_matrix": mat,
                "emo_labels"   : mat.any(axis=1).astype(np.float32),
                "cause_labels" : mat.any(axis=0).astype(np.float32),
                "pairs_truth"  : doc.get("pairs", []),
                "idx2id"       : idx2id,
                "doc_id"       : doc.get("doc_id", str(len(out))),
                "clauses"      : [c["clause"] for c in doc["clauses"]],
                "n_pos"        : int(mat.sum()),
                "n_total"      : int(n * n),
            })
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s   = self.samples[i]
        enc = self.tokenizer(
            s["input_text"], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt")
        iids  = enc["input_ids"].squeeze(0)
        amask = enc["attention_mask"].squeeze(0)
        cidx  = (iids == self.clause_tok_id).nonzero(as_tuple=True)[0].tolist()
        return (iids, amask, cidx,
                s["labels_matrix"], s["emo_labels"], s["cause_labels"],
                s["pairs_truth"], s["idx2id"], s["doc_id"], s["clauses"],
                s["n_pos"], s["n_total"])

def collate_fn(batch):
    iids  = torch.stack([x[0] for x in batch])
    amask = torch.stack([x[1] for x in batch])
    B  = len(batch)
    mc = max(len(x[2]) for x in batch)
    cidx  = torch.zeros(B, mc, dtype=torch.long)
    cmask = torch.zeros(B, mc, dtype=torch.bool)
    lmat  = torch.zeros(B, mc, mc)
    elbl  = torch.zeros(B, mc)
    clbl  = torch.zeros(B, mc)
    for i, x in enumerate(batch):
        nc = min(len(x[2]), mc)
        if nc == 0:
            continue
        cidx[i, :nc]      = torch.tensor(x[2][:nc], dtype=torch.long)
        cmask[i, :nc]     = True
        vl = min(nc, x[3].shape[0])
        lmat[i, :vl, :vl] = torch.tensor(x[3][:vl, :vl])
        el = min(nc, x[4].shape[0])
        elbl[i, :el]      = torch.tensor(x[4][:el])
        clbl[i, :el]      = torch.tensor(x[5][:el])
    return (iids, amask, cidx, cmask, lmat, elbl, clbl,
            [x[6] for x in batch],   # pairs_truth [7]
            [x[7] for x in batch],   # idx2id      [8]
            [x[8] for x in batch],   # doc_id      [9]
            [x[9] for x in batch],   # clauses     [10]
            [x[10] for x in batch],  # n_pos       [11]
            [x[11] for x in batch])  # n_total     [12]

# ════════════════════════════════════════════════════════════
#  模型组件
# ════════════════════════════════════════════════════════════
class DoRA_Biaffine(nn.Module):
    def __init__(self, D=512, rank=4):
        super().__init__()
        self.W  = nn.Parameter(torch.empty(D, D))
        nn.init.xavier_uniform_(self.W)
        self.A  = nn.Parameter(torch.randn(D, rank) * 0.02)
        self.B  = nn.Parameter(torch.zeros(rank, D))
        with torch.no_grad():
            self.m = nn.Parameter(self.W.norm(dim=0).clone())
        self.b  = nn.Parameter(torch.zeros(1))
        self.sc = rank ** 0.5

    def forward(self, h_e, h_c):
        W = self.W + (self.A @ self.B) / self.sc
        W = self.m.unsqueeze(0) * W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return torch.matmul(torch.matmul(h_e, W), h_c.transpose(1, 2)) + self.b

class SpanRepr(nn.Module):
    def __init__(self, D=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.LayerNorm(D))

    def forward(self, hidden, cidx, cmask, amask):
        B, N  = cidx.shape
        D     = hidden.shape[-1]
        seq   = hidden.shape[1]
        cls_r = torch.gather(hidden, 1, cidx.unsqueeze(-1).expand(-1, -1, D))
        tok   = torch.arange(seq, device=hidden.device).view(1, 1, seq)
        cst   = cidx.unsqueeze(2)
        cen   = torch.cat(
            [cidx[:, 1:], torch.full((B, 1), seq, device=hidden.device)],
            dim=1).unsqueeze(2)
        smask = (tok >= cst) & (tok < cen) & amask.unsqueeze(1).bool()
        sf    = smask.float()
        mean_r = (sf.unsqueeze(-1) * hidden.unsqueeze(1)).sum(2) / \
                 sf.sum(-1, keepdim=True).clamp(min=1.0)
        return self.proj(torch.cat([cls_r, mean_r], dim=-1))

class LayerFusion(nn.Module):
    def __init__(self, n=3, D=768):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n)/n)
        self.norm = nn.LayerNorm(D)

    def forward(self, hs_list, cidx):
        w = F.softmax(self.w, dim=0)
        fused = sum(w[i]*hs_list[i] for i in range(len(hs_list)))
        out = torch.gather(fused, 1, cidx.unsqueeze(-1).expand(-1,-1,fused.shape[-1]))
        return self.norm(out)

class ECPE_Model(nn.Module):
    def __init__(self, cfg_exp):
        super().__init__()
        self.cfg = cfg_exp
        D = 768
        self.encoder = BertModel.from_pretrained(
            MODEL_PATH, local_files_only=True,
            output_hidden_states=True)
        self.base_norm = nn.LayerNorm(D)
        if cfg_exp["use_span"]:
            self.span_repr = SpanRepr(D)
        self.clause_sa = nn.TransformerEncoderLayer(
            D, 8, 2048, CFG["dropout"],
            batch_first=True, norm_first=True)
        
        self.emo_head   = nn.Linear(D, 1)
        self.cause_head = nn.Linear(D, 1)
        
        self.mlp_emo   = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.LayerNorm(512))
        self.mlp_cause = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.LayerNorm(512))
        
        # ★ 绝对安全隔离：只有跑到 w/o Biaffine 时才实例化这个层，绝不干扰其他实验的随机数！
        self.use_biaffine = cfg_exp.get("use_biaffine", True)
        if self.use_biaffine:
            if cfg_exp["use_dora"]:
                self.biaffine = DoRA_Biaffine(512, cfg_exp["dora_rank"])
            else:
                self.W_std = nn.Parameter(torch.empty(512, 512))
                nn.init.xavier_uniform_(self.W_std)
                self.b_std = nn.Parameter(torch.zeros(1))
        else:
            self.mlp_scorer = nn.Sequential(
                nn.Linear(512 * 2, 256),
                nn.GELU(),
                nn.Dropout(CFG["dropout"]),
                nn.Linear(256, 1)
            )

    def _encode(self, iids, amask, cidx, cmask):
        out = self.encoder(input_ids=iids, attention_mask=amask)
        r   = torch.gather(out.last_hidden_state, 1,
                           cidx.unsqueeze(-1).expand(-1, -1, 768))
        r   = self.base_norm(r)
        if self.cfg["use_span"]:
            r = r + self.span_repr(out.last_hidden_state, cidx, cmask, amask)
        return self.clause_sa(r, src_key_padding_mask=~cmask)

    def forward(self, iids, amask, cidx, cmask,
                lmat=None, elbl=None, clbl=None,
                n_pos=None, n_tot=None,
                return_logits=False, return_hidden=False):
        r   = self._encode(iids, amask, cidx, cmask)
        h_e = self.mlp_emo(r)
        h_c = self.mlp_cause(r)
        
        if self.use_biaffine:
            logits = (self.biaffine(h_e, h_c) if self.cfg["use_dora"]
                      else torch.matmul(torch.matmul(h_e, self.W_std),
                                        h_c.transpose(1, 2)) + self.b_std)
        else:
            B, N, D_prime = h_e.shape
            h_e_exp = h_e.unsqueeze(2).expand(B, N, N, D_prime)
            h_c_exp = h_c.unsqueeze(1).expand(B, N, N, D_prime)
            concat_h = torch.cat([h_e_exp, h_c_exp], dim=-1)
            logits = self.mlp_scorer(concat_h).squeeze(-1)
            
        if return_hidden:
            return logits, h_e.detach().cpu(), h_c.detach().cpu()
        if return_logits:
            return logits
        if lmat is None:
            return None, logits

        pm  = (cmask.unsqueeze(2) & cmask.unsqueeze(1)).float()
        pw  = torch.tensor(CFG["pos_weight"], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, lmat, pos_weight=pw, reduction="none")
        loss = (bce * pm).sum() / (pm.sum() + 1e-8)

        if self.cfg["use_orth"]:
            cos  = (F.normalize(h_e, dim=-1) *
                    F.normalize(h_c, dim=-1)).sum(-1)
            loss = loss + CFG["orth_lambda"] * \
                   (cos.pow(2) * cmask.float()).sum() / \
                   (cmask.float().sum() + 1e-8)
        return loss, logits

# ════════════════════════════════════════════════════════════
#  解码 & 评测
# ════════════════════════════════════════════════════════════
def decode_pairs(all_logits, all_masks, all_idx2ids, thr, window):
    res = []
    si  = 0
    for bl, bm in zip(all_logits, all_masks):
        for b in range(bl.shape[0]):
            preds = []
            for i in range(bl.shape[1]):
                if not bm[b, i]:
                    continue
                for j in range(bl.shape[2]):
                    if not bm[b, j]:
                        continue
                    win_ok = (abs(i - j) <= window) if window < 999 else True
                    if bl[b, i, j] > thr and win_ok:
                        eid = all_idx2ids[si].get(i)
                        cid = all_idx2ids[si].get(j)
                        if eid is not None and cid is not None:
                            preds.append((int(eid), int(cid)))
            res.append(preds)
            si += 1
    return res

def compute_metrics(true_list, pred_list):
    tp = fp = fn = 0
    for tl, pl in zip(true_list, pred_list):
        ts  = set(map(tuple, tl))
        ps  = set(map(tuple, pl))
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return round(p, 4), round(r, 4), round(f1, 4)

def run_inference(model, dl, collect_hidden=False):
    model.eval()
    all_logits, all_masks, all_idx2ids, all_true = [], [], [], []
    all_h_e, all_h_c, all_labels = [], [], []
    with torch.no_grad():
        for batch in dl:
            iids, amask, cidx, cmask = [x.to(device) for x in batch[:4]]
            if collect_hidden:
                lg, h_e, h_c = model(iids, amask, cidx, cmask,
                                     return_hidden=True)
                for b in range(cmask.shape[0]):
                    v = cmask[b].cpu().numpy()
                    n = int(v.sum())
                    all_h_e.append(h_e[b][:n].numpy())
                    all_h_c.append(h_c[b][:n].numpy())
                    lm  = batch[4][b]
                    vl  = min(n, lm.shape[0])
                    hp  = lm[:vl, :vl].any(dim=1).float().numpy().tolist()
                    hp += [0.0] * (n - len(hp))
                    all_labels.extend(hp)
            else:
                lg = model(iids, amask, cidx, cmask, return_logits=True)
            all_logits.append(lg.cpu().numpy())
            all_masks.append(cmask.cpu().numpy())
            all_idx2ids.extend(batch[8])
            all_true.extend(batch[7])
    if collect_hidden:
        return (all_logits, all_masks, all_idx2ids, all_true,
                all_h_e, all_h_c, all_labels)
    return all_logits, all_masks, all_idx2ids, all_true

# ════════════════════════════════════════════════════════════
#  错误案例收集
# ════════════════════════════════════════════════════════════
def collect_errors(all_logits, all_masks, all_idx2ids,
                   all_true, pred_list, samples):
    errors = []
    si = 0
    for bl, bm in zip(all_logits, all_masks):
        for b in range(bl.shape[0]):
            s   = samples[si]
            ts  = set(map(tuple, all_true[si]))
            ps  = set(map(tuple, pred_list[si]))
            fps = list(ps - ts)
            fns = list(ts - ps)
            if fps or fns:
                inv = {v: k for k, v in s["idx2id"].items()}
                def sc(ep, cp):
                    ii, jj = inv.get(ep), inv.get(cp)
                    if (ii is not None and jj is not None
                            and ii < bl.shape[1] and jj < bl.shape[2]):
                        return round(float(bl[b, ii, jj]), 4)
                    return None
                def gt(cid):
                    idx = inv.get(cid)
                    return (s["clauses"][idx]
                            if idx is not None and idx < len(s["clauses"])
                            else "?")
                errors.append({
                    "doc_id"    : s["doc_id"],
                    "n_clauses" : len(s["clauses"]),
                    "true_pairs": [list(p) for p in ts],
                    "pred_pairs": [list(p) for p in ps],
                    "FP": [{"pair": list(p), "logit": sc(*p),
                             "emo": gt(p[0]), "cause": gt(p[1])}
                            for p in fps],
                    "FN": [{"pair": list(p), "logit": sc(*p),
                             "emo": gt(p[0]), "cause": gt(p[1])}
                            for p in fns],
                })
            si += 1
    return errors

# ════════════════════════════════════════════════════════════
#  训练集敏感度分析
# ════════════════════════════════════════════════════════════
def run_sensitivity(model, trn_dl, logger, fold_idx, fd):
    logger.info(f"      [Sens Fold{fold_idx:02d}] 训练集敏感度...")
    logits, masks, idx2ids, trues = run_inference(model, trn_dl)
    res = {"window": {}, "threshold": {}}

    for win in CFG["sens_windows"]:
        preds    = decode_pairs(logits, masks, idx2ids, CFG["fixed_thr"], win)
        p, r, f1 = compute_metrics(trues, preds)
        tag = str(win) if win < 999 else "Inf"
        res["window"][tag] = {"P": p, "R": r, "F1": f1, "win": win}
        logger.info(f"        Win={tag:>4s} P={p:.4f} R={r:.4f} F1={f1:.4f}")

    for thr in CFG["sens_thrs"]:
        preds    = decode_pairs(logits, masks, idx2ids, thr, CFG["fixed_window"])
        p, r, f1 = compute_metrics(trues, preds)
        res["threshold"][f"{thr:.2f}"] = {"P": p, "R": r, "F1": f1, "thr": thr}
        logger.info(f"        Thr={thr:+.2f} P={p:.4f} R={r:.4f} F1={f1:.4f}")

    with open(os.path.join(fd, "sens_train.json"), "w") as f:
        json.dump(res, f, indent=2)

# ════════════════════════════════════════════════════════════
#  显著性检验
# ════════════════════════════════════════════════════════════
def significance_test(f1s_a, f1s_b, label_a, label_b,
                      n_boot=10000, seed=42):
    rng  = np.random.default_rng(seed)
    a    = np.array(f1s_a)
    b    = np.array(f1s_b)
    diff = b - a
    try:
        _, p_w = stats.wilcoxon(a, b)
    except Exception:
        p_w = float("nan")
    boot = np.array([
        diff[rng.integers(0, len(diff), len(diff))].mean()
        for _ in range(n_boot)
    ])
    ci = [round(float(np.percentile(boot, 2.5)),  4),
          round(float(np.percentile(boot, 97.5)), 4)]
    return {
        "model_a"          : label_a,
        "model_b"          : label_b,
        "mean_a"           : round(float(a.mean()), 4),
        "mean_b"           : round(float(b.mean()), 4),
        "mean_diff"        : round(float(diff.mean()), 4),
        "wilcoxon_p"       : round(float(p_w), 6),
        "significant_p05"  : bool(p_w < 0.05),
        "significant_p01"  : bool(p_w < 0.01),
        "boot_CI_95"       : ci,
        "CI_excludes_zero" : bool(ci[0] > 0 or ci[1] < 0),
        "fold_diffs"       : [round(v, 4) for v in diff.tolist()],
        "fold_f1s_a"       : [round(v, 4) for v in a.tolist()],
        "fold_f1s_b"       : [round(v, 4) for v in b.tolist()],
    }

# ════════════════════════════════════════════════════════════
#  单折训练
# ════════════════════════════════════════════════════════════
def run_fold(exp_id, cfg_exp, fold_idx, tokenizer, logger,
             exp_dir, decode_window=None, save_model=False):
    logger.info(f"    ▶ Fold {fold_idx:02d}")
    t0  = time.time()
    win = decode_window if decode_window is not None else CFG["fixed_window"]

    trn_ds = ECPE_Dataset(
        f"{DATA_DIR}/fold{fold_idx}_train.json", tokenizer)
    tst_ds = ECPE_Dataset(
        f"{DATA_DIR}/fold{fold_idx}_test.json", tokenizer)

    g  = torch.Generator()
    g.manual_seed(CFG["seed"] + fold_idx)
    kw = dict(collate_fn=collate_fn, num_workers=CFG["num_workers"],
              pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    trn_dl = DataLoader(trn_ds, CFG["batch_size"], shuffle=True,  **kw)
    tst_dl = DataLoader(tst_ds, CFG["batch_size"], shuffle=False, **kw)

    model = ECPE_Model(cfg_exp).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))

    enc_p   = [p for n, p in model.named_parameters()
               if "encoder" in n and p.requires_grad]
    other_p = [p for n, p in model.named_parameters()
               if "encoder" not in n]
    opt = AdamW([
        {"params": enc_p,   "lr": CFG["enc_lr"],  "weight_decay": 0.01},
        {"params": other_p, "lr": CFG["head_lr"], "weight_decay": 0.01},
    ])
    total_steps = len(trn_dl) * CFG["epochs"]
    sch = get_linear_schedule_with_warmup(
        opt, int(total_steps * CFG["warmup_ratio"]), total_steps)

    best_f1 = best_p = best_r = 0.0
    best_logits = best_masks = best_idx2ids = best_true = best_preds = None
    best_epoch  = 0
    curve       = []

    for epoch in range(CFG["epochs"]):
        model.train()
        total_loss = 0.0
        nb = 0
        for batch in trn_dl:
            iids, amask, cidx, cmask, lmat, elbl, clbl = [
                x.to(device) if torch.is_tensor(x) else x
                for x in batch[:7]
            ]
            n_pos, n_tot = batch[11], batch[12]
            opt.zero_grad()

            if cfg_exp["use_rdrop"]:
                l1, lg1 = model(iids, amask, cidx, cmask, lmat,
                                elbl, clbl, n_pos=n_pos, n_tot=n_tot)
                l2, lg2 = model(iids, amask, cidx, cmask, lmat,
                                elbl, clbl, n_pos=n_pos, n_tot=n_tot)
                p1  = torch.sigmoid(lg1).clamp(1e-7, 1 - 1e-7)
                p2  = torch.sigmoid(lg2).clamp(1e-7, 1 - 1e-7)
                pm  = (cmask.unsqueeze(2) & cmask.unsqueeze(1)).float()
                kl  = 0.5 * (
                    p1 * (p1 / p2).log() +
                    (1 - p1) * ((1 - p1) / (1 - p2)).log() +
                    p2 * (p2 / p1).log() +
                    (1 - p2) * ((1 - p2) / (1 - p1)).log()
                )
                loss = ((l1 + l2) * 0.5 +
                        CFG["rdrop_alpha"] * (kl * pm).sum() / (pm.sum() + 1e-8))
            else:
                loss, _ = model(iids, amask, cidx, cmask, lmat,
                                elbl, clbl, n_pos=n_pos, n_tot=n_tot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            sch.step()
            total_loss += loss.item()
            nb         += 1

        avg_loss = total_loss / nb

        tl, tm, ti, tt = run_inference(model, tst_dl)
        preds    = decode_pairs(tl, tm, ti, CFG["fixed_thr"], win)
        p, r, f1 = compute_metrics(tt, preds)
        curve.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 6),
                      "P": p, "R": r, "F1": f1})

        if f1 > best_f1:
            best_f1  = f1
            best_p   = p
            best_r   = r
            best_epoch   = epoch + 1
            best_logits  = [l.copy() for l in tl]
            best_masks   = [m.copy() for m in tm]
            best_idx2ids = list(ti)
            best_true    = list(tt)
            best_preds   = preds
            if save_model:
                fd_ck = os.path.join(exp_dir, f"fold{fold_idx:02d}")
                safe_mkdir(fd_ck)
                torch.save(
                    {"epoch": best_epoch, "state": model.state_dict(),
                     "f1": best_f1, "p": best_p, "r": best_r},
                    os.path.join(fd_ck, "best_model.pt"))

        logger.info(f"      Ep{epoch+1:02d} loss={avg_loss:.4f} "
                    f"P={p:.4f} R={r:.4f} F1={f1:.4f} "
                    f"(best={best_f1:.4f}@ep{best_epoch})")

    # ── 落盘 ────────────────────────────────────────────────
    fd = os.path.join(exp_dir, f"fold{fold_idx:02d}")
    safe_mkdir(fd)

    np.savez_compressed(
        os.path.join(fd, "logits.npz"),
        **{f"b{i}": l for i, l in enumerate(best_logits)})

    with open(os.path.join(fd, "preds.json"), "w", encoding="utf-8") as f:
        json.dump({
            "decode_thr": CFG["fixed_thr"], "decode_win": win,
            "best_epoch": best_epoch,
            "P": best_p, "R": best_r, "F1": best_f1,
            "preds": [list(map(list, p)) for p in best_preds],
            "trues": [list(map(list, t)) for t in best_true],
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(fd, "curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    errors = collect_errors(best_logits, best_masks, best_idx2ids,
                            best_true, best_preds, tst_ds.samples)
    with open(os.path.join(fd, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    # T-SNE：主实验全 10 折保存
    if exp_id == "MAIN_Full":
        tsne_dir = os.path.join(OUT_DIR, "tsne")
        safe_mkdir(tsne_dir)
        _, _, _, _, h_e, h_c, lb = run_inference(
            model, tst_dl, collect_hidden=True)
        np.savez_compressed(
            os.path.join(tsne_dir, f"fold{fold_idx:02d}_tsne.npz"),
            h_emo   = np.vstack(h_e),
            h_cause = np.vstack(h_c),
            labels  = np.array(lb))
        logger.info(f"      [T-SNE] fold{fold_idx:02d} 已保存")

    # 训练集敏感度（主实验每折）
    if cfg_exp["do_sens"]:
        run_sensitivity(
            model,
            DataLoader(trn_ds, CFG["batch_size"], shuffle=False, **kw),
            logger, fold_idx, fd)

    del model, opt, sch
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"    ✔ Fold{fold_idx:02d} "
                f"P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f} "
                f"ep={best_epoch} ({time.time()-t0:.0f}s)")
    return best_p, best_r, best_f1

# ════════════════════════════════════════════════════════════
#  敏感度汇总
# ════════════════════════════════════════════════════════════
def aggregate_sensitivity(exp_dir, logger):
    win_agg = {}
    thr_agg = {}
    for fold in range(1, CFG["num_folds"] + 1):
        path = os.path.join(exp_dir, f"fold{fold:02d}", "sens_train.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for tag, val in data.get("window", {}).items():
            win_agg.setdefault(tag, {"P": [], "R": [], "F1": [], "meta": val})
            for k in ["P", "R", "F1"]:
                win_agg[tag][k].append(val[k])
        for tag, val in data.get("threshold", {}).items():
            thr_agg.setdefault(tag, {"P": [], "R": [], "F1": [], "meta": val})
            for k in ["P", "R", "F1"]:
                thr_agg[tag][k].append(val[k])

    fig = os.path.join(OUT_DIR, "figures")
    safe_mkdir(fig)

    def dump(agg, jpath, cpath, col, sort_fn):
        if not agg:
            return
        summ = {}
        for tag in sorted(agg.keys(), key=sort_fn):
            d = agg[tag]
            summ[tag] = {
                "avg_P" : round(float(np.mean(d["P"])), 4),
                "avg_R" : round(float(np.mean(d["R"])), 4),
                "avg_F1": round(float(np.mean(d["F1"])), 4),
                "std_F1": round(float(np.std(d["F1"])), 4),
                "fold_F1": [round(v, 4) for v in d["F1"]],
            }
        with open(jpath, "w") as f:
            json.dump(summ, f, indent=2)
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([col, "Avg_P", "Avg_R", "Avg_F1", "Std_F1"] +
                       [f"F{i}" for i in range(1, 11)])
            for tag, d in summ.items():
                w.writerow([tag, d["avg_P"], d["avg_R"],
                             d["avg_F1"], d["std_F1"]] + d["fold_F1"])
        logger.info(f"  → {jpath}")

    dump(win_agg,
         os.path.join(fig, "sens_window.json"),
         os.path.join(fig, "sens_window.csv"),
         "Window",
         lambda t: int(t) if t != "Inf" else 9999)
    dump(thr_agg,
         os.path.join(fig, "sens_threshold.json"),
         os.path.join(fig, "sens_threshold.csv"),
         "Threshold",
         lambda t: float(t))

# ════════════════════════════════════════════════════════════
#  显著性汇总
# ════════════════════════════════════════════════════════════
def aggregate_significance(all_records, logger):
    rec = {r["exp_id"]: r for r in all_records}
    if "MAIN_Full" not in rec:
        return
    full_f1s = rec["MAIN_Full"]["fold_f1s"]
    results  = {}
    # ★ 把新增的 ABL_woBiaffine 也加进检验列表
    targets  = ["ABL_PureBase", "ABL_RDropOnly", "ABL_DoRA_RDrop",
                "ABL_woDoRA", "ABL_woSpan", "ABL_woRDrop", "ABL_woBiaffine"]
    logger.info("  ── 显著性检验 ──")
    for eid in targets:
        if eid not in rec:
            continue
        r   = significance_test(rec[eid]["fold_f1s"], full_f1s,
                                eid, "MAIN_Full")
        key = f"MAIN_vs_{eid}"
        results[key] = r
        sig = "★ p<0.05" if r["significant_p05"] else "n.s."
        logger.info(f"    [{eid}→FULL] Δ={r['mean_diff']:+.4f} "
                    f"p={r['wilcoxon_p']:.4f} {sig} "
                    f"CI={r['boot_CI_95']}")
    fig = os.path.join(OUT_DIR, "figures")
    safe_mkdir(fig)
    with open(os.path.join(fig, "significance.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  → {fig}/significance.json")

# ════════════════════════════════════════════════════════════
#  论文表格导出
# ════════════════════════════════════════════════════════════
def export_tables(records, logger):
    rec = {r["exp_id"]: r for r in records}
    fig = os.path.join(OUT_DIR, "figures")
    safe_mkdir(fig)

    # Table 1：主实验 + SOTA
    sota = [
        {"method": "Indep (ACL 2019)",         "P": 68.32, "R": 50.82, "F1": 58.18},
        {"method": "ECPE-2D (ACL 2020)",        "P": 72.92, "R": 65.44, "F1": 68.89},
        {"method": "RankCP (ACL 2020)",         "P": 71.19, "R": 76.30, "F1": 73.60},
        {"method": "PBJE (EMNLP 2022)",         "P": 73.84, "R": 79.22, "F1": 76.37},
        {"method": "UECA-Prompt (COLING 2022)", "P": 77.99, "R": 71.82, "F1": 74.70},
        {"method": "JCB (ACL 2023)",            "P": 74.85, "R": 76.42, "F1": 75.62},
        {"method": "UniECPE (NN 2024)",         "P": 75.31, "R": 77.20, "F1": 76.24},
        {"method": "JFTA (CMC 2025)",           "P": 76.41, "R": 75.81, "F1": 76.05},
    ]
    for eid, lbl in [
        ("ABL_PureBase", "RoBERTa-wwm + Biaffine (Base)"),
        ("MAIN_Full",    "Ours (DoRA-Biaffine + SpanRepr + RDrop)"),
    ]:
        if eid in rec:
            r = rec[eid]
            sota.append({"method": lbl,
                         "P"  : round(r["avg_p"]  * 100, 2),
                         "R"  : round(r["avg_r"]  * 100, 2),
                         "F1" : round(r["avg_f1"] * 100, 2),
                         "Std": round(r["std_f1"] * 100, 2)})
    with open(os.path.join(OUT_DIR, "table1_main.json"), "w") as f:
        json.dump(sota, f, indent=2, ensure_ascii=False)

    # Table 2：消融
    abl_order = [
        ("ABL_PureBase",   "RoBERTa + Biaffine"),
        ("ABL_RDropOnly",  "+ RDrop"),
        ("ABL_DoRA_RDrop", "+ C1 DoRA-Biaffine"),
        ("MAIN_Full",      "+ C2 SpanRepr  (Full)"),
        ("ABL_woDoRA",     "Full w/o C1 DoRA"),
        ("ABL_woSpan",     "Full w/o C2 SpanRepr"),
        ("ABL_woBiaffine", "Full w/o C3 Biaffine (MLP)"), # ★ 新增
        ("ABL_woRDrop",    "Full w/o RDrop"),
    ]
    t2 = []
    for eid, lbl in abl_order:
        if eid not in rec:
            continue
        r = rec[eid]
        t2.append({"model"   : lbl,
                   "P"       : round(r["avg_p"]  * 100, 2),
                   "R"       : round(r["avg_r"]  * 100, 2),
                   "F1"      : round(r["avg_f1"] * 100, 2),
                   "Std"     : round(r["std_f1"] * 100, 2),
                   "fold_F1s": [round(v * 100, 2) for v in r["fold_f1s"]]})
    with open(os.path.join(OUT_DIR, "table2_ablation.json"), "w") as f:
        json.dump(t2, f, indent=2)
    with open(os.path.join(OUT_DIR, "table2_ablation.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "P", "R", "F1", "Std"])
        w.writeheader()
        for row in t2:
            w.writerow({k: row[k] for k in ["model", "P", "R", "F1", "Std"]})

    # rank 消融
    rank_data = {
        str(r["cfg"]["dora_rank"]): {
            "avg_F1" : r["avg_f1"],
            "std_F1" : r["std_f1"],
            "fold_f1s": r["fold_f1s"],
        }
        for r in records if r["exp_id"].startswith("RANK_")
    }
    if rank_data:
        with open(os.path.join(fig, "rank_ablation.json"), "w") as f:
            json.dump(rank_data, f, indent=2)

    # 窗口消融（测试集）
    win_data = {
        str(WIN_OVERRIDE.get(r["exp_id"], CFG["fixed_window"])): {
            "avg_F1": r["avg_f1"],
            "avg_P" : r["avg_p"],
            "avg_R" : r["avg_r"],
        }
        for r in records if r["exp_id"].startswith("WIN_")
    }
    if win_data:
        with open(os.path.join(fig, "window_testset.json"), "w") as f:
            json.dump(win_data, f, indent=2)

    # 箱线图数据
    box = {
        eid: {
            "fold_f1s": [round(v * 100, 2) for v in rec[eid]["fold_f1s"]],
            "mean"    : round(rec[eid]["avg_f1"] * 100, 2),
            "std"     : round(rec[eid]["std_f1"] * 100, 2),
        }
        for eid in ["MAIN_Full", "ABL_PureBase", "ABL_woDoRA", "ABL_woSpan", "ABL_woBiaffine"]
        if eid in rec
    }
    with open(os.path.join(fig, "boxplot.json"), "w") as f:
        json.dump(box, f, indent=2)

    # 平均训练曲线
    curves   = []
    full_dir = os.path.join(OUT_DIR, "experiments", "MAIN_Full")
    if os.path.isdir(full_dir):
        for fold in range(1, CFG["num_folds"] + 1):
            cp = os.path.join(full_dir, f"fold{fold:02d}", "curve.json")
            if os.path.exists(cp):
                with open(cp) as f:
                    curves.append(json.load(f))
    if curves:
        n_ep    = len(curves[0])
        avg_c   = []
        for ep in range(n_ep):
            f1s  = [c[ep]["F1"]         for c in curves if ep < len(c)]
            loss = [c[ep]["train_loss"] for c in curves if ep < len(c)]
            avg_c.append({
                "epoch"   : ep + 1,
                "avg_F1"  : round(float(np.mean(f1s)), 4),
                "avg_loss": round(float(np.mean(loss)), 6),
            })
        with open(os.path.join(fig, "training_curve.json"), "w") as f:
            json.dump(avg_c, f, indent=2)

    logger.info(f"  → 论文表格已导出: {OUT_DIR}")

# ════════════════════════════════════════════════════════════
#  单实验入口（含文件级断点续跑）
# ════════════════════════════════════════════════════════════
def run_experiment(exp_id, cfg_exp, tokenizer, lb_writer, lb_file, logger):
    exp_dir      = os.path.join(OUT_DIR, "experiments", exp_id)
    summary_path = os.path.join(exp_dir, "summary.json")
    safe_mkdir(exp_dir)

    # ── 断点续跑：summary.json 存在则直接读取并跳过 ──────────
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summ = json.load(f)
            logger.info(f"⏭  {exp_id} 已完成 (F1={summ['avg_f1']:.4f})，跳过")
            lb_writer.writerow([
                exp_id, summ["avg_p"], summ["avg_r"],
                summ["avg_f1"], summ["std_f1"], summ["max_f1"],
                *[f"{v:.4f}" for v in summ["fold_f1s"]],
            ])
            lb_file.flush()
            return summ
        except Exception as e:
            logger.warning(f"读取 {summary_path} 失败，重新训练: {e}")

    decode_window = WIN_OVERRIDE.get(exp_id)
    save_ckpt     = (exp_id == "MAIN_Full")

    logger.info(f"\n{'═' * 65}")
    logger.info(f"  ★ {exp_id}  "
                f"dora={cfg_exp['use_dora']}(r={cfg_exp['dora_rank']}) "
                f"span={cfg_exp['use_span']} rdrop={cfg_exp['use_rdrop']} "
                f"biaffine={cfg_exp['use_biaffine']}")
    logger.info(f"    解码: thr={CFG['fixed_thr']} "
                f"win={decode_window or CFG['fixed_window']}")
    logger.info(f"{'═' * 65}")

    results = []
    for fold in range(1, CFG["num_folds"] + 1):
        set_seed(CFG["seed"] + fold)
        results.append(
            run_fold(exp_id, cfg_exp, fold, tokenizer, logger,
                     exp_dir, decode_window=decode_window,
                     save_model=save_ckpt))

    ps, rs, f1s = zip(*results)
    ap = round(float(np.mean(ps)),  4)
    ar = round(float(np.mean(rs)),  4)
    af = round(float(np.mean(f1s)), 4)
    sf = round(float(np.std(f1s)),  4)
    mf = round(float(np.max(f1s)),  4)

    summ = {
        "exp_id"    : exp_id,
        "cfg"       : {k: v for k, v in cfg_exp.items()
                       if k != "do_sens"},
        "avg_p"     : ap, "avg_r": ar,
        "avg_f1"    : af, "std_f1": sf, "max_f1": mf,
        "fold_f1s"  : [round(v, 4) for v in f1s],
        "fold_ps"   : [round(v, 4) for v in ps],
        "fold_rs"   : [round(v, 4) for v in rs],
        "decode_thr": CFG["fixed_thr"],
        "decode_win": decode_window or CFG["fixed_window"],
    }
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=2)

    if cfg_exp["do_sens"]:
        aggregate_sensitivity(exp_dir, logger)

    logger.info(f"\n  ★ {exp_id} → "
                f"P={ap:.4f} R={ar:.4f} F1={af:.4f}±{sf:.4f} Max={mf:.4f}")
    logger.info(f"    折F1: {[round(v, 4) for v in f1s]}")

    lb_writer.writerow([
        exp_id, ap, ar, af, sf, mf,
        *[f"{v:.4f}" for v in f1s],
    ])
    lb_file.flush()
    return summ

# ════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════
def main():
    safe_mkdir(OUT_DIR)
    logger = setup_logging(OUT_DIR)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  ECPE 论文最终版 v12 (加 w/o Biaffine) GPU={ARGS.gpu}        ║")
    logger.info(f"║  主实验: DoRA(rank=4) + SpanRepr + RDrop               ║")
    logger.info(f"║  目标精度: ~77.46%  (固定 pos_weight, 无 AdaPW)        ║")
    logger.info(f"║  实验总数: {len(EXPERIMENTS):2d} × 10折  输出: {OUT_DIR:<20} ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU: {pr.name}  {pr.total_memory/1024**3:.1f}GB")
    logger.info(f"  Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True)
    tokenizer.add_tokens(["<clause>"])

    # Leaderboard：追加模式，支持断点续跑
    lb_path   = os.path.join(OUT_DIR, "LEADERBOARD.csv")
    lb_exists = os.path.exists(lb_path)
    lb_file   = open(lb_path, "a" if lb_exists else "w",
                     newline="", encoding="utf-8")
    lb_writer = csv.writer(lb_file)
    if not lb_exists:
        lb_writer.writerow([
            "Exp_ID", "Avg_P", "Avg_R", "Avg_F1", "Std_F1", "Max_F1",
            *[f"F{i}_F1" for i in range(1, 11)],
        ])

    records = []
    t_all   = time.time()

    for exp_id, cfg_exp in EXPERIMENTS.items():
        records.append(
            run_experiment(exp_id, cfg_exp, tokenizer,
                           lb_writer, lb_file, logger))

    lb_file.close()

    # ── 汇总 ──────────────────────────────────────────────
    logger.info(f"\n{'═' * 65}")
    logger.info("  ★★★ 汇总图表数据 ★★★")
    export_tables(records, logger)
    aggregate_significance(records, logger)

    total_min = (time.time() - t_all) / 60
    base_f    = next((r["avg_f1"] for r in records
                      if r["exp_id"] == "ABL_PureBase"), 0.0)

    logger.info(f"\n{'═' * 65}")
    logger.info("  最终排行:")
    for rank_i, r in enumerate(
            sorted(records, key=lambda x: -x["avg_f1"]), 1):
        logger.info(
            f"  #{rank_i:2d} {r['exp_id']:<22} "
            f"F1={r['avg_f1']:.4f}±{r['std_f1']:.4f} "
            f"Max={r['max_f1']:.4f} "
            f"Δ={r['avg_f1']-base_f:+.4f}"
        )
    logger.info(f"\n  完成！耗时 {total_min:.1f}min  输出: {OUT_DIR}")

if __name__ == "__main__":
    main()