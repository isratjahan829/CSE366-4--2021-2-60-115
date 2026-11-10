# %% [markdown]
# # HyperDiag: Hypergraph Attention for Symptom-Based Differential Diagnosis
#
# **Thesis pre-defence experiments: full, executed pipeline**
#
# Dataset: *Diseases and Symptoms Dataset* (Kaggle, `dhivyeshrk/diseases-and-symptoms-dataset`),
# file `Final_Augmented_dataset_Diseases_and_Symptoms.csv` (246,945 rows × 1 label + 377 binary symptoms, 773 diseases).
#
# This notebook implements every step of the methodology chapter of the draft and adds three new contributions
# that came out of auditing the data:
#
# | # | Contribution | Where |
# |---|---|---|
# | C1 | **Data audit and leakage-free protocol.** 23% of rows are exact duplicates, and many symptom vectors map to more than one disease. Random splits therefore leak test cases into training. We propose a *symptom-vector-grouped stratified split* and derive the **Bayes-optimal top-k ceiling** of the task. | §2, §3 |
# | C2 | **HyperDiag**: disease hyperedges plus co-occurrence hyperedges; **dual-level attention** (node→hyperedge, hyperedge→node); **query-aware hyperedge attention**; and **disease-hyperedge prototype scoring**, where each class is scored by its own learned hyperedge embedding. | §4, §5 |
# | C3 | **Clinically motivated evaluation**: top-k differential ranking (Top-1/3/5/10, MRR, macro-F1), **partial-symptom robustness**, **rare-disease analysis**, an **ablation study**, **statistical significance**, and an **attention-faithfulness (deletion) test**. | §6–§10 |
#
# Every number in the result tables is produced by this notebook. The draft's placeholder tables are filled from `results/*.csv`.

# %% [markdown]
# ## ⚙️ User settings: the only cell you need to edit
# Set `DATASET` to **any one** of the following, then choose **Run All**:
# * a Kaggle dataset link, e.g. `https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset`
#   (on Kaggle you can also attach it via **+ Add Input**; the notebook finds it under `/kaggle/input` automatically)
# * a Kaggle slug, e.g. `dhivyeshrk/diseases-and-symptoms-dataset`
# * a direct link to a `.csv` or `.zip` file
# * a local path to the CSV file or to the folder that contains it
#
# `FAST = True` runs a quick check (4 epochs, 1 seed, about 10 min). `FAST = False` runs the full experiment
# (20 epochs, 3 seeds; about 80 min on CPU, much faster on a Kaggle GPU).

# %%
DATASET = "https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset"
FAST = False

# %% [markdown]
# ## 0. Setup and configuration

# %%
import os, json, time, math, random, warnings, platform
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.stats import spearmanr, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

FAST = os.environ.get("HYPERDIAG_FAST", "1" if FAST else "0") == "1"   # set in the user-settings cell
CFG = dict(
    min_cases=5,          # diseases with fewer unique cases cannot be split train/val/test reliably
    test_frac=0.20, val_frac=0.10,
    dim=128, heads=4, layers=2, dropout=0.2,
    batch=1024, lr=2e-3, weight_decay=1e-4,
    epochs=4 if FAST else 20, patience=100,
    seeds=[0] if FAST else [0, 1, 2],
    sym_dropout=float(os.environ.get("SYMDROP", 0.5)),  # max per-case symptom-masking rate (all neural models)
    label_smoothing=0.05,
    cw_power=0.5,         # class weight ~ n_c^-0.5 (imbalance handling)
    dis_thresh=0.02,      # P(s|d) threshold for disease-hyperedge membership
    cooc_k=4, cooc_min_npmi=0.10, cooc_min_count=10,
    graph_k=10,           # neighbours per symptom in the pairwise graph (GCN/GAT)
    keep_ratios=[1.0, 0.8, 0.6, 0.4, 0.2],
)
ROOT = os.path.abspath(".")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def savefig(name):
    plt.savefig(os.path.join(FIG, name)); plt.show(); plt.close("all")

print("python", platform.python_version(), "| torch", torch.__version__, "| device", DEV, "| threads", torch.get_num_threads())
print("FAST mode" if FAST else "FULL mode", json.dumps(CFG))

# %% [markdown]
# ## 1. Dataset collection
# `DATASET` (from the user-settings cell) is resolved automatically: local path, then `/kaggle/input`, then a download
# (with `kagglehub` for Kaggle links, or a plain HTTP download for direct links).

# %%
import glob, re, zipfile, shutil, urllib.request
DATASET = os.environ.get("HYPERDIAG_DATA", DATASET)
CSV_NAME = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"

def pick_csv(folder):
    """Return the dataset CSV inside a folder (prefers the known file name, else the largest CSV)."""
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not files: return None
    named = [f for f in files if os.path.basename(f) == CSV_NAME]
    return named[0] if named else max(files, key=os.path.getsize)

def resolve_dataset(spec):
    spec = str(spec).strip()
    if os.path.isfile(spec): return spec
    if os.path.isdir(spec):
        f = pick_csv(spec)
        if f: return f
    # Kaggle notebook with the dataset attached as input
    if os.path.isdir("/kaggle/input"):
        f = pick_csv("/kaggle/input")
        if f: return f
    # previously downloaded copy
    f = pick_csv("data") if os.path.isdir("data") else None
    if f: return f
    m = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", spec)
    slug = m.group(1) if m else (spec if re.fullmatch(r"[\w.-]+/[\w.-]+", spec) else None)
    if slug:
        try:
            import kagglehub
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
            import kagglehub
        f = pick_csv(kagglehub.dataset_download(slug))
        if f: return f
    if spec.startswith("http"):
        os.makedirs("data", exist_ok=True)
        dst = os.path.join("data", os.path.basename(spec.split("?")[0]) or "dataset.csv")
        print("downloading", spec); urllib.request.urlretrieve(spec, dst)
        if zipfile.is_zipfile(dst):
            zipfile.ZipFile(dst).extractall("data"); return pick_csv("data")
        return dst
    raise FileNotFoundError(f"Could not find the dataset from DATASET = {spec!r}. "
                            "Set DATASET to a Kaggle link, a direct CSV/ZIP link, or a local path.")

DATA_PATH = resolve_dataset(DATASET)
header = pd.read_csv(DATA_PATH, nrows=0).columns
raw = pd.read_csv(DATA_PATH, dtype={c: np.uint8 for c in header if c != "diseases"})
print("file:", DATA_PATH)
print("shape:", raw.shape)
raw.iloc[:5, :12]

# %% [markdown]
# ## 2. Data audit (Contribution C1)
# Before modelling we check the properties that decide how the task should be evaluated:
# missing values, binarity, duplicates, *conflicting* symptom vectors (same symptoms, different disease), class imbalance and unused symptoms.

# %%
LABEL = "diseases"
raw[LABEL] = raw[LABEL].astype(str).str.strip().str.lower()
Xraw = raw.drop(columns=LABEL).values
sym_names_all = np.array(raw.columns.drop(LABEL))
vec_key_raw = pd.util.hash_pandas_object(raw.drop(columns=LABEL), index=False).values

audit = {}
audit["rows"] = len(raw)
audit["diseases"] = raw[LABEL].nunique()
audit["symptoms"] = Xraw.shape[1]
audit["missing_values"] = int(raw.isna().sum().sum())
audit["binary_only"] = bool(set(np.unique(Xraw)) <= {0, 1})
audit["rows_with_zero_symptoms"] = int((Xraw.sum(1) == 0).sum())
audit["exact_duplicate_rows"] = int(raw.duplicated().sum())
audit["unique_symptom_vectors"] = int(pd.Series(vec_key_raw).nunique())
g = pd.DataFrame({"k": vec_key_raw, "y": raw[LABEL].values})
ndis_per_vec = g.groupby("k")["y"].nunique()
audit["vectors_mapping_to_>1_disease"] = int((ndis_per_vec > 1).sum())
audit["rows_with_ambiguous_vector"] = int(g["k"].map(ndis_per_vec).gt(1).sum())
audit["unused_symptoms"] = int((Xraw.sum(0) == 0).sum())
vc_raw = raw[LABEL].value_counts()
audit["max_class"], audit["median_class"], audit["min_class"] = int(vc_raw.max()), float(vc_raw.median()), int(vc_raw.min())
audit["imbalance_ratio"] = float(vc_raw.max() / vc_raw.min())
audit["symptoms_per_case_mean"] = float(Xraw.sum(1).mean())
audit_df = pd.DataFrame(audit.items(), columns=["property", "value"])
audit_df.to_csv(os.path.join(RES, "data_audit.csv"), index=False)
audit_df

# %% [markdown]
# ### 2.1 Bayes-optimal ceiling
# When the same symptom vector appears with several diseases, no classifier can tell those cases apart.
# For each distinct vector $x$ with disease counts $n_{x,d}$, the best achievable top-k hit rate is
# $\frac{1}{N}\sum_x \sum_{d\in \text{top-}k(n_{x,\cdot})} n_{x,d}$. We compute this ceiling on the raw data and on the de-duplicated data.

# %%
def bayes_ceiling(keys, labels, ks=(1, 3, 5, 10)):
    t = pd.DataFrame({"k": keys, "y": labels}).value_counts().rename("n").reset_index()
    t = t.sort_values(["k", "n"], ascending=[True, False])
    t["r"] = t.groupby("k").cumcount() + 1
    N = t["n"].sum()
    return {f"Top-{k}": t.loc[t.r <= k, "n"].sum() / N for k in ks}

dedup_tmp = raw.drop_duplicates()
ceil_raw = bayes_ceiling(vec_key_raw, raw[LABEL].values)
ceil_dedup = bayes_ceiling(pd.util.hash_pandas_object(dedup_tmp.drop(columns=LABEL), index=False).values, dedup_tmp[LABEL].values)
ceiling_df = pd.DataFrame([ceil_raw, ceil_dedup], index=["raw data", "de-duplicated"]).round(4)
ceiling_df.to_csv(os.path.join(RES, "bayes_ceiling.csv"))
ceiling_df

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
ax[0].plot(np.arange(len(vc_raw)), vc_raw.values, color="#3b6ea8"); ax[0].set_yscale("log")
ax[0].set(title="Class distribution (773 diseases, sorted)", xlabel="disease rank", ylabel="#cases (log)")
k = Xraw.sum(1); ax[1].hist(k, bins=np.arange(0.5, k.max() + 1.5), color="#3b6ea8", edgecolor="white")
ax[1].set(title="Symptoms per case", xlabel="#symptoms present", ylabel="#cases")
top = pd.Series(Xraw.sum(0), index=sym_names_all).sort_values(ascending=False).head(15)[::-1]
ax[2].barh(top.index, top.values, color="#3b6ea8"); ax[2].set(title="15 most frequent symptoms", xlabel="#cases")
savefig("fig_eda.png")

# %% [markdown]
# ## 3. Preprocessing, missing values, imbalance and leakage-free split
# 1. Standardise labels (done above) and remove **exact duplicate records**.
# 2. Missing or malformed indicators would be set to 0 (absent). The audit shows there are none.
# 3. Drop **unused symptoms** (never present).
# 4. Keep diseases with at least `min_cases` unique cases so each disease can appear in train, validation and test.
# 5. **Leakage-free split**: a stratified *group* split where the *group is the symptom vector*. No symptom pattern in the
#    test set is ever seen during training. The split stays stratified by disease.

# %%
df = raw.drop_duplicates().reset_index(drop=True)
X_all = df.drop(columns=LABEL).values
used = X_all.sum(0) > 0
vc = df[LABEL].value_counts()
keep_dis = vc[vc >= CFG["min_cases"]].index
df = df[df[LABEL].isin(keep_dis)].reset_index(drop=True)
X = df.drop(columns=LABEL).values[:, used].astype(np.float32)
sym_names = sym_names_all[used]
classes = np.array(sorted(df[LABEL].unique()))
cls2id = {c: i for i, c in enumerate(classes)}
y = df[LABEL].map(cls2id).values
groups = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
S, C = X.shape[1], len(classes)
print(f"after preprocessing: {len(df):,} cases | {S} symptoms | {C} diseases "
      f"(dropped {audit['diseases'] - C} diseases with < {CFG['min_cases']} unique cases, "
      f"{audit['unused_symptoms']} unused symptoms)")

def grouped_split(y, groups, seed=42):
    """Stratified *group* split: every distinct symptom vector (group) goes to exactly one of train/val/test.
    Groups are stratified by their (majority) disease, so class proportions are preserved."""
    g = pd.DataFrame({"g": groups, "y": y})
    gl = g.groupby("g")["y"].agg(lambda s: s.value_counts().index[0])          # majority disease of each group
    gid, glab = gl.index.values, gl.values
    cnt = pd.Series(glab).map(pd.Series(glab).value_counts()).values
    strat = np.where(cnt >= 10, glab, -1)                                       # very rare strata pooled
    g_trv, g_te = train_test_split(gid, test_size=CFG["test_frac"], random_state=seed, stratify=strat)
    s2 = pd.Series(strat, index=gid)[g_trv].values
    g_tr, g_va = train_test_split(g_trv, test_size=CFG["val_frac"] / (1 - CFG["test_frac"]), random_state=seed, stratify=s2)
    where = pd.Series(0, index=gid); where[g_va] = 1; where[g_te] = 2
    w = where.reindex(groups).values
    return np.flatnonzero(w == 0), np.flatnonzero(w == 1), np.flatnonzero(w == 2)

t0 = time.time()
tr_idx, va_idx, te_idx = grouped_split(y, groups)
print(f"split in {time.time()-t0:.1f}s -> train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,}")
overlap = len(set(groups[te_idx]) & set(groups[tr_idx]))
print("symptom vectors shared by train and test:", overlap)
miss = set(range(C)) - set(y[tr_idx])
print("diseases absent from train:", len(miss), "| diseases present in test:", len(set(y[te_idx])))

# %% [markdown]
# ### 3.1 How much do common evaluation protocols leak?
# We compare three protocols with the same simple MLP:
# **(A)** random split on raw data (the usual practice), **(B)** random split after de-duplication, **(C)** our grouped split.
# *Leak* is the fraction of test cases whose exact symptom vector also occurs in training.

# %%
def mlp_quick(Xtr, ytr, Xte, yte, ncls, epochs=8, seed=0):
    set_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, ncls)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 512):
            b = perm[i:i + 512]; opt.zero_grad()
            F.cross_entropy(m(Xt[b].to(DEV)), yt[b].to(DEV)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lo = torch.cat([m(torch.tensor(Xte[i:i + 4096]).to(DEV)).cpu() for i in range(0, len(Xte), 4096)])
    rank = (lo > lo.gather(1, torch.tensor(yte)[:, None])).sum(1) + 1
    return (rank == 1).float().mean().item(), (rank <= 5).float().mean().item()

leak_rows = []
# (A) raw, random split
ra = raw[raw[LABEL].isin(keep_dis)].reset_index(drop=True)
XA = ra.drop(columns=LABEL).values[:, used].astype(np.float32); yA = ra[LABEL].map(cls2id).values
gA = pd.util.hash_pandas_object(pd.DataFrame(XA), index=False).values
a_tr, a_te = train_test_split(np.arange(len(yA)), test_size=0.2, random_state=42, stratify=yA)
# (B) de-duplicated, random split
b_tr, b_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
for name, (Xp, yp, gp, itr, ite) in {
        "(A) random split, raw data": (XA, yA, gA, a_tr, a_te),
        "(B) random split, de-duplicated": (X, y, groups, b_tr, b_te),
        "(C) grouped split (ours)": (X, y, groups, np.concatenate([tr_idx, va_idx]), te_idx)}.items():
    trset = set(gp[itr]); leak = np.mean([k in trset for k in gp[ite]])
    t1, t5 = mlp_quick(Xp[itr], yp[itr], Xp[ite], yp[ite], C)
    ce = bayes_ceiling(gp[ite], yp[ite], ks=(1,))["Top-1"]
    leak_rows.append(dict(protocol=name, n_test=len(ite), leaked_test_cases=round(leak, 4),
                          MLP_top1=round(t1, 4), MLP_top5=round(t5, 4), test_top1_ceiling=round(ce, 4)))
leak_df = pd.DataFrame(leak_rows); leak_df.to_csv(os.path.join(RES, "leakage_protocols.csv"), index=False)
leak_df

# %% [markdown]
# ### 3.2 Class-imbalance handling
# Neural models use a **class-weighted** cross-entropy with $w_c \propto n_c^{-0.5}$ (normalised to mean 1) and mild label smoothing.
# Training also uses **variable-rate symptom-masking augmentation**: each case draws its own rate $r\sim U(0, r_{max})$ and drops each present symptom with probability $r$, always keeping at least one.
# This simulates patients who report only some of their symptoms and acts as resampling at the feature level.
# We use re-weighting rather than oversampling because exact copies of rare cases would recreate the duplicate problem found in §2.

# %%
Xtr, ytr = X[tr_idx], y[tr_idx]; Xva, yva = X[va_idx], y[va_idx]; Xte, yte = X[te_idx], y[te_idx]
n_c = np.bincount(ytr, minlength=C).astype(np.float64)
cw = (np.maximum(n_c, 1) ** -CFG["cw_power"]); cw = cw / cw.mean()
class_w = torch.tensor(cw, dtype=torch.float32)
print("train class counts: min", int(n_c.min()), "median", int(np.median(n_c)), "max", int(n_c.max()))
print("class weight range: %.2f – %.2f" % (cw.min(), cw.max()))

# %% [markdown]
# ## 4. Hypergraph construction (training data only)
# **Nodes**: the $|V|$ symptoms.
#
# **Disease hyperedges** $E_d$: one per disease $d$, containing every symptom with $P(s\mid d) \ge \tau$. The incidence weight is $h_{sd}=P(s\mid d)$.
#
# **Co-occurrence hyperedges** $E_c$: for every symptom $v$, the set $\{v\}\cup$ its top-$k$ partners by normalised PMI
# (NPMI ≥ 0.10, co-count ≥ 10), de-duplicated. The weight is the mean NPMI. These capture symptom clusters that recur *across* diseases.
#
# **Pairwise graph** (for the GCN/GAT baselines): the same NPMI statistics, but with only the top-10 pairwise edges per symptom.
# This is the pairwise counterpart of $E_c$.
#
# Node features $X^0$ combine a learnable identity embedding with a projection of the symptom's PPMI co-occurrence profile.

# %%
def build_structures(Xtr, ytr):
    Xs = sparse.csr_matrix(Xtr)
    # --- disease hyperedges
    Y = sparse.csr_matrix((np.ones(len(ytr)), (ytr, np.arange(len(ytr)))), shape=(C, len(ytr)))
    cnt = (Y @ Xs).toarray()                                # C x S counts
    p_sd = cnt / np.maximum(np.bincount(ytr, minlength=C)[:, None], 1)        # P(s|d)
    Hd = (p_sd >= CFG["dis_thresh"]) & (cnt >= 1)
    for d in np.where(Hd.sum(1) == 0)[0]:                   # safety: every disease gets a hyperedge
        Hd[d, np.argsort(-p_sd[d])[:3]] = True
    Wd = np.where(Hd, p_sd, 0.0)
    # --- co-occurrence statistics
    co = (Xs.T @ Xs).toarray().astype(np.float64); f = np.diag(co).copy(); N = len(ytr)
    pij = co / N; pi = f / N
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(pij / np.outer(pi, pi)); npmi = pmi / (-np.log(pij))
    npmi[~np.isfinite(npmi)] = -1; np.fill_diagonal(npmi, -1)
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0); np.fill_diagonal(ppmi, 0)
    ok = (npmi >= CFG["cooc_min_npmi"]) & (co >= CFG["cooc_min_count"])
    sets = {}
    for v in range(S):
        nb = [u for u in np.argsort(-npmi[v]) if ok[v, u]][:CFG["cooc_k"]]
        if nb:
            key = tuple(sorted([v] + nb))
            sets[key] = np.mean([npmi[v, u] for u in nb])
    Hc = np.zeros((len(sets), S), bool); Wc = np.zeros((len(sets), S))
    for i, (key, w) in enumerate(sets.items()):
        Hc[i, list(key)] = True; Wc[i, list(key)] = w
    # --- pairwise kNN graph (GCN / GAT)
    A = np.zeros((S, S), bool)
    for v in range(S):
        nb = [u for u in np.argsort(-npmi[v]) if ok[v, u]][:CFG["graph_k"]]
        A[v, nb] = True
    A = A | A.T; np.fill_diagonal(A, True)
    return dict(p_sd=p_sd, cnt=cnt, n_d=np.bincount(ytr, minlength=C), Hd=Hd, Wd=Wd, Hc=Hc, Wc=Wc, npmi=npmi, ppmi=ppmi, A=A, freq=f)

t0 = time.time(); ST = build_structures(Xtr, ytr)
print(f"built in {time.time()-t0:.1f}s")
hstats = pd.DataFrame({
    "hyperedge type": ["disease", "co-occurrence"],
    "count": [ST["Hd"].shape[0], ST["Hc"].shape[0]],
    "mean size": [ST["Hd"].sum(1).mean(), ST["Hc"].sum(1).mean()],
    "median size": [np.median(ST["Hd"].sum(1)), np.median(ST["Hc"].sum(1))],
    "max size": [ST["Hd"].sum(1).max(), ST["Hc"].sum(1).max()],
}).round(2)
node_deg = ST["Hd"].sum(0) + ST["Hc"].sum(0)
print("pairwise graph edges (undirected, excl. self):", int((ST["A"].sum() - S) / 2))
print("symptom node degree in hypergraph: mean %.1f, min %d, max %d" % (node_deg.mean(), node_deg.min(), node_deg.max()))
hstats.to_csv(os.path.join(RES, "hypergraph_stats.csv"), index=False); hstats

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.4]})
ax[0].hist(ST["Hd"].sum(1), bins=30, color="#3b6ea8", edgecolor="white"); ax[0].set(title="Disease-hyperedge size", xlabel="#symptoms per disease")
ax[1].hist(node_deg, bins=30, color="#d07a3a", edgecolor="white"); ax[1].set(title="Symptom node degree", xlabel="#hyperedges containing the symptom")
ex = np.argsort(-n_c)[:12:2]
cols = np.where(ST["Hd"][ex].any(0))[0]
cols = cols[np.argsort(-ST["Wd"][ex][:, cols].max(0))][:25]
sns.heatmap(ST["Wd"][ex][:, cols], ax=ax[2], cmap="Blues", xticklabels=sym_names[cols], yticklabels=classes[ex], cbar_kws={"label": "P(s|d)"})
ax[2].set_title("Incidence matrix excerpt (disease hyperedges)"); ax[2].tick_params(axis="x", labelsize=7, rotation=90); ax[2].tick_params(axis="y", labelsize=8)
savefig("fig_hypergraph.png")

# %% [markdown]
# ## 5. Models
# All neural models share the same training recipe (optimiser, class-weighted loss, augmentation, early stopping on validation MRR).
# **Non-neural baselines:** Bernoulli Naive Bayes, Random Forest (size-capped so it fits in memory with 600+ classes), softmax Logistic Regression.
# **Neural baselines:** MLP; GCN and GAT on the pairwise symptom graph; HGNN (Feng et al., 2019), a hypergraph convolution without attention.
#
# **HyperDiag (proposed)**, with $L$ stacked dual-attention hypergraph layers:
# * *Node-level attention* (node → hyperedge): $\alpha_{ve}=\mathrm{softmax}_{v\in e}\big(\mathrm{LReLU}(a^\top[W x_v \Vert U c_e]) + \log h_{ve}\big)$, $\;c_e \leftarrow \sum_v \alpha_{ve} W x_v$.
# * *Hyperedge-level attention* (hyperedge → node): $\beta_{ev}=\mathrm{softmax}_{e\ni v}\big(\mathrm{LReLU}(b^\top[c_e \Vert W x_v])\big)$, $\;x_v \leftarrow \mathrm{LN}(x_v+\sigma(\sum_e \beta_{ev} c_e))$.
# * *Patient query*: context-aware attention pooling over the patient's present symptoms ($\gamma_v$ is the symptom-evidence weight used for explanations).
# * *Query-aware hyperedge attention*: the query attends over co-occurrence hyperedges that touch its symptoms and refines itself with the matched symptom clusters.
# * *Disease-hyperedge prototype scoring*: $s_d = \langle \phi(q), \psi(c_d) \rangle/\tau + b_d$. Each disease is scored by its *own* hyperedge embedding, so structure is tied directly to the decision.
# * *Hyperedge-likelihood evidence with query-adaptive gating*: each disease hyperedge has soft incidence $h_{vd}=P(v\mid d)$ (Laplace-smoothed).
#   The final score adds generative evidence from the hyperedge,
#   $s_d \mathrel{+}= \lambda_+(q)\sum_{v\in S}\log h_{vd} + \lambda_-(q)\sum_{v\notin S}\log(1-h_{vd})$,
#   where $\lambda_\pm=\mathrm{softplus}(g([q;\log(1+|S|)]))$ are predicted *per patient*. When few symptoms are reported, the model can learn
#   that missing symptoms carry little information, which pure discriminative models cannot express.

# %%
def masked_softmax(logits, mask, dim):
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim) * mask

class SetAttnReadout(nn.Module):
    """Context-aware attention pooling over the symptoms present in a case."""
    def __init__(self, d):
        super().__init__(); self.W = nn.Linear(d, d); self.U = nn.Linear(d, d, bias=False); self.a = nn.Linear(d, 1, bias=False)
    def forward(self, Hn, M):                 # Hn: S x d, M: B x S (0/1)
        cnt = M.sum(1, keepdim=True).clamp(min=1)
        ctx = (M @ Hn) / cnt                  # B x d mean of present symptoms
        # score_v = a^T tanh(W h_v)  +  <U ctx, h_v>/sqrt(d): symptom salience + agreement with the rest of the case
        sc = self.a(torch.tanh(self.W(Hn))).squeeze(-1)[None] + (self.U(ctx) @ Hn.T) / math.sqrt(Hn.shape[1])
        g = masked_softmax(sc, M > 0, 1)
        return g @ Hn, ctx, g

class NodeInit(nn.Module):
    def __init__(self, d, ppmi):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(S, d) * 0.1)
        P = torch.tensor(ppmi, dtype=torch.float32); self.register_buffer("P", P / (P.norm(dim=1, keepdim=True) + 1e-8))
        self.proj = nn.Linear(S, d, bias=False)
    def forward(self): return self.emb + self.proj(self.P)

class Head(nn.Module):
    def __init__(self, din, d, drop):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, d * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 2, C))
    def forward(self, z): return self.net(z)

class LogReg(nn.Module):
    def __init__(self, **k): super().__init__(); self.lin = nn.Linear(S, C)
    def forward(self, M): return self.lin(M)

class MLP(nn.Module):
    def __init__(self, d=128, drop=0.2, **k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S, 4 * d), nn.LayerNorm(4 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(4 * d, 4 * d), nn.LayerNorm(4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, C))
    def forward(self, M): return self.net(M)

class GraphReadoutModel(nn.Module):
    """Shared shell for GCN / GAT / HGNN: node encoder -> attention readout -> MLP head."""
    def __init__(self, d, drop, ppmi):
        super().__init__(); self.init = NodeInit(d, ppmi); self.read = SetAttnReadout(d); self.head = Head(2 * d, d, drop); self.drop = nn.Dropout(drop)
    def encode(self): raise NotImplementedError
    def forward(self, M, return_att=False):
        Hn = self.encode(); z, ctx, g = self.read(Hn, M)
        out = self.head(torch.cat([z, ctx], 1))
        return (out, dict(gamma=g)) if return_att else out

class GCN(GraphReadoutModel):
    def __init__(self, d=128, layers=2, drop=0.2, ppmi=None, A=None, **k):
        super().__init__(d, drop, ppmi)
        A = torch.tensor(A, dtype=torch.float32); dinv = A.sum(1).rsqrt()
        self.register_buffer("Ah", dinv[:, None] * A * dinv[None])
        self.lins = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)]); self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
    def encode(self):
        h = self.init()
        for lin, ln in zip(self.lins, self.lns): h = ln(h + self.drop(F.gelu(self.Ah @ lin(h))))
        return h

class GATLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.h, self.dh = heads, d // heads
        self.W = nn.Linear(d, d, bias=False); self.a_s = nn.Parameter(torch.randn(heads, self.dh) * 0.1); self.a_d = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
    def forward(self, x, mask):
        z = self.W(x).view(-1, self.h, self.dh)
        e = F.leaky_relu((z * self.a_d).sum(-1)[:, None] + (z * self.a_s).sum(-1)[None], 0.2)  # S x S x h
        att = masked_softmax(e, mask[:, :, None], 1)
        return torch.einsum("ijh,jhd->ihd", att, z).reshape(-1, self.h * self.dh)

class GAT(GraphReadoutModel):
    def __init__(self, d=128, layers=2, heads=4, drop=0.2, ppmi=None, A=None, **k):
        super().__init__(d, drop, ppmi); self.register_buffer("mask", torch.tensor(A))
        self.layers = nn.ModuleList([GATLayer(d, heads) for _ in range(layers)]); self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
    def encode(self):
        h = self.init()
        for lay, ln in zip(self.layers, self.lns): h = ln(h + self.drop(F.elu(lay(h, self.mask))))
        return h

def incidence(ST, use_cooc=True):
    Hm = ST["Hd"].T if not use_cooc else np.concatenate([ST["Hd"], ST["Hc"]]).T   # S x E
    Hw = ST["Wd"].T if not use_cooc else np.concatenate([ST["Wd"], ST["Wc"]]).T
    return torch.tensor(Hm), torch.tensor(Hw, dtype=torch.float32)

class HGNN(GraphReadoutModel):
    """Feng et al. (2019): X' = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Θ (no attention)."""
    def __init__(self, d=128, layers=2, drop=0.2, ppmi=None, ST=None, **k):
        super().__init__(d, drop, ppmi)
        Hm, Hw = incidence(ST); H = Hm.float()
        dv = H.sum(1).clamp(min=1); de = H.sum(0).clamp(min=1)
        self.register_buffer("G", (dv.rsqrt()[:, None] * H / de[None]) @ (H.T * dv.rsqrt()[None]))
        self.lins = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)]); self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
    def encode(self):
        h = self.init()
        for lin, ln in zip(self.lins, self.lns): h = ln(h + self.drop(F.gelu(self.G @ lin(h))))
        return h

class DualAttnHyperLayer(nn.Module):
    def __init__(self, d, heads, drop, node_attn=True, edge_attn=True):
        super().__init__(); self.h, self.dh = heads, d // heads; self.node_attn, self.edge_attn = node_attn, edge_attn
        self.Wv = nn.Linear(d, d, bias=False); self.Ue = nn.Linear(d, d, bias=False)
        self.a_v = nn.Parameter(torch.randn(heads, self.dh) * 0.1); self.a_e = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.b_e = nn.Parameter(torch.randn(heads, self.dh) * 0.1); self.b_v = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.out_e = nn.Linear(d, d); self.out_v = nn.Linear(d, d)
        self.ln_e = nn.LayerNorm(d); self.ln_v = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x, c, Hm, logw, Hw_norm, Hdeg_norm):
        S_, E_ = Hm.shape
        z = self.Wv(x).view(S_, self.h, self.dh); ce = self.Ue(c).view(E_, self.h, self.dh)
        m3 = Hm[:, :, None]
        # node -> hyperedge (softmax over member nodes of each hyperedge)
        if self.node_attn:
            lg = F.leaky_relu((z * self.a_v).sum(-1)[:, None] + (ce * self.a_e).sum(-1)[None], 0.2) + logw[:, :, None]
            alpha = masked_softmax(lg, m3, 0)                       # S x E x h
        else:
            alpha = Hw_norm[:, :, None].expand(-1, -1, self.h)
        e_msg = torch.einsum("seh,shd->ehd", alpha, z)
        c_new = self.ln_e(c + self.drop(F.elu(self.out_e(e_msg.reshape(E_, -1)))))
        # hyperedge -> node (softmax over hyperedges incident to each node)
        if self.edge_attn:
            lg2 = F.leaky_relu((e_msg * self.b_e).sum(-1)[None] + (z * self.b_v).sum(-1)[:, None], 0.2)
            beta = masked_softmax(lg2, m3, 1)                       # S x E x h
        else:
            beta = Hdeg_norm[:, :, None].expand(-1, -1, self.h)
        v_msg = torch.einsum("seh,ehd->shd", beta, e_msg)
        x_new = self.ln_v(x + self.drop(F.elu(self.out_v(v_msg.reshape(S_, -1)))))
        return x_new, c_new, alpha, beta

class HyperDiag(nn.Module):
    def __init__(self, d=128, layers=2, heads=4, drop=0.2, ppmi=None, ST=None,
                 use_cooc=True, node_attn=True, edge_attn=True, proto=True, query_edge_attn=True, hyper_prior=True, **k):
        super().__init__()
        # hyperedge-likelihood evidence: smoothed soft incidence h_vd = P(v | disease-hyperedge d)
        h = (ST["cnt"] + 0.5) / (ST["n_d"][:, None] + 1.0)
        self.register_buffer("log_h", torch.tensor(np.log(h).T, dtype=torch.float32))        # S x C
        self.register_buffer("log_1mh", torch.tensor(np.log1p(-h).T, dtype=torch.float32))   # S x C
        self.hyper_prior = hyper_prior
        self.gate = nn.Linear(d + 1, 2)                       # query-adaptive weights of present / absent evidence
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -3.0)   # start with weak evidence
        Hm, Hw = incidence(ST, use_cooc)
        self.register_buffer("Hm", Hm); self.register_buffer("logw", torch.log(Hw.clamp(min=1e-6)))
        self.register_buffer("Hw_norm", Hw / Hw.sum(0, keepdim=True).clamp(min=1e-9))
        Hf = Hm.float(); self.register_buffer("Hdeg_norm", Hf / Hf.sum(1, keepdim=True).clamp(min=1))
        self.nD = C; self.nE = Hm.shape[1]
        self.init = NodeInit(d, ppmi)
        self.etype = nn.Parameter(torch.randn(2, d) * 0.1)
        self.register_buffer("is_cooc", torch.arange(self.nE) >= C)
        self.layers = nn.ModuleList([DualAttnHyperLayer(d, heads, drop, node_attn, edge_attn) for _ in range(layers)])
        self.read = SetAttnReadout(d)
        self.use_cooc, self.qea, self.proto = use_cooc, query_edge_attn and use_cooc, proto
        self.q_att = nn.Linear(d, d, bias=False); self.q_upd = nn.Linear(d, d)
        self.qproj = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        if proto:
            self.pproj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.bias = nn.Parameter(torch.zeros(C)); self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        else:
            self.head = Head(d, d, drop)
        self.drop = nn.Dropout(drop)

    def encode(self):
        x = self.init()
        c = (self.Hw_norm.T @ x) + self.etype[self.is_cooc.long()]
        atts = []
        for lay in self.layers:
            x, c, a, b = lay(x, c, self.Hm, self.logw, self.Hw_norm, self.Hdeg_norm); atts.append((a, b))
        return x, c, atts

    def forward(self, M, return_att=False):
        x, c, atts = self.encode()
        z, ctx, g = self.read(x, M)
        q = self.qproj(torch.cat([z, ctx], 1))
        q_alpha = None
        if self.qea:
            cc = c[C:]
            touch = (M @ self.Hm[:, C:].float()) > 0                  # co-occ. hyperedges touching the query
            sc = (self.q_att(q) @ cc.T) / math.sqrt(q.shape[1])
            q_alpha = masked_softmax(sc, touch, 1)
            q = q + self.drop(self.q_upd(q_alpha @ cc))
        if self.proto:
            P = self.pproj(c[:C])
            out = self.logit_scale.exp() * (F.normalize(q, dim=1) @ F.normalize(P, dim=1).T) + self.bias
        else:
            out = self.head(q)
        if self.hyper_prior:
            lam = F.softplus(self.gate(torch.cat([q, torch.log1p(M.sum(1, keepdim=True))], 1)))  # B x 2
            ev = lam[:, :1] * (M @ self.log_h) + lam[:, 1:] * ((1 - M) @ self.log_1mh)
            out = out + ev
        return (out, dict(gamma=g, layer_att=atts, q_alpha=q_alpha)) if return_att else out

print("models defined")

# %% [markdown]
# ### 5.1 Training and evaluation utilities
# Metrics: **Top-k accuracy** (k = 1, 3, 5, 10) and **MRR**, both computed from the full ranking over all diseases,
# plus **macro-F1** and weighted-F1 of the top-1 prediction over the diseases present in the test set.

# %%
def augment(M, p):
    """Variable-rate symptom masking: each case gets its own drop rate ~ U(0, p)."""
    if p <= 0: return M
    rate = torch.rand(M.shape[0], 1, device=M.device) * p
    drop = (torch.rand_like(M) < rate) & (M > 0)
    Mn = M.masked_fill(drop, 0.0)
    empty = Mn.sum(1) == 0
    if empty.any(): Mn[empty] = M[empty]
    return Mn

@torch.no_grad()
def predict_logits(model, Xm, bs=4096):
    model.eval()
    return torch.cat([model(torch.as_tensor(Xm[i:i + bs]).to(DEV)).float().cpu() for i in range(0, len(Xm), bs)])

def rank_metrics(scores, yt, ks=(1, 3, 5, 10)):
    scores = torch.as_tensor(scores, dtype=torch.float32); yt_t = torch.as_tensor(yt)
    true = scores.gather(1, yt_t[:, None])
    rank = ((scores > true).sum(1) + 1).numpy()
    pred = scores.argmax(1).numpy()
    out = {f"Top-{k}": float((rank <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / rank).mean())
    labs = np.unique(yt)
    out["Macro-F1"] = float(f1_score(yt, pred, labels=labs, average="macro", zero_division=0))
    out["Weighted-F1"] = float(f1_score(yt, pred, labels=labs, average="weighted", zero_division=0))
    return out, rank, pred

def train_neural(builder, seed, Xtr, ytr, Xva, yva, verbose=True, tag=""):
    set_seed(seed); model = builder().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=CFG["epochs"] * math.ceil(len(ytr) / CFG["batch"]), pct_start=0.1)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr); w = class_w.to(DEV)
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(CFG["epochs"]):
        model.train(); perm = torch.randperm(len(yt)); tot = 0.0
        for i in range(0, len(yt), CFG["batch"]):
            b = perm[i:i + CFG["batch"]]
            M = augment(Xt[b].to(DEV), CFG["sym_dropout"])
            loss = F.cross_entropy(model(M), yt[b].to(DEV), weight=w, label_smoothing=CFG["label_smoothing"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.item() * len(b)
        m, _, _ = rank_metrics(predict_logits(model, Xva), yva)
        hist.append(dict(epoch=ep + 1, loss=tot / len(yt), **{f"val_{k}": v for k, v in m.items()}))
        if m["MRR"] > best + 1e-5:
            best, bad = m["MRR"], 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or bad == 0 and ep == CFG["epochs"] - 1):
            print(f"  [{tag} s{seed}] ep{ep+1:02d} loss {tot/len(yt):.4f} val top1 {m['Top-1']:.4f} top5 {m['Top-5']:.4f} MRR {m['MRR']:.4f} ({time.time()-t0:.0f}s)")
        if bad >= CFG["patience"]: break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), time.time() - t0

def partial_mask(Xm, r, seed=123):
    rs = np.random.RandomState(seed); out = np.zeros_like(Xm)
    for i, row in enumerate(Xm):
        on = np.flatnonzero(row); keep = max(1, int(math.ceil(r * len(on))))
        out[i, rs.choice(on, keep, replace=False)] = 1
    return out
PART = {r: (Xte if r == 1.0 else partial_mask(Xte, r)) for r in CFG["keep_ratios"]}   # shared partial-symptom test sets

def show(x):
    try: display(x)
    except NameError: print(x)

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

common = dict(d=CFG["dim"], layers=CFG["layers"], heads=CFG["heads"], drop=CFG["dropout"], ppmi=ST["ppmi"], A=ST["A"], ST=ST)
BUILDERS = {
    "LogReg (softmax)": lambda: LogReg(**common),
    "MLP": lambda: MLP(**common),
    "GCN (pairwise)": lambda: GCN(**common),
    "GAT (pairwise)": lambda: GAT(**common),
    "HGNN (hypergraph, no attention)": lambda: HGNN(**common),
    "HyperDiag (ours)": lambda: HyperDiag(**common),
}
for n, b in BUILDERS.items(): print(f"{n:34s} params = {nparams(b()):,}")

# %% [markdown]
# ## 6. Experiments: baselines and HyperDiag
# ### 6.1 Classical machine-learning baselines

# %%
results, store = [], {}   # store[(model, seed)] = test logits
t0 = time.time()
nb = BernoulliNB(alpha=0.1).fit(sparse.csr_matrix(Xtr), ytr)
s_nb = nb.predict_log_proba(sparse.csr_matrix(Xte)); m, _, _ = rank_metrics(s_nb, yte)
results.append(dict(model="Naive Bayes (Bernoulli)", seed=0, time_s=time.time() - t0, **m)); store[("Naive Bayes (Bernoulli)", 0)] = s_nb
print("NB", {k: round(v, 4) for k, v in m.items()}, f"{time.time()-t0:.0f}s")

t0 = time.time()
rf = RandomForestClassifier(n_estimators=60, max_leaf_nodes=4000, n_jobs=-1, random_state=0).fit(Xtr, ytr)
s_rf = np.concatenate([rf.predict_proba(Xte[i:i + 5000]) for i in range(0, len(Xte), 5000)])
s_rf = s_rf + 1e-9 * np.random.RandomState(0).rand(*s_rf.shape)        # tie-break for ranking
m, _, _ = rank_metrics(s_rf, yte)
results.append(dict(model="Random Forest", seed=0, time_s=time.time() - t0, **m)); store[("Random Forest", 0)] = s_rf
print("RF", {k: round(v, 4) for k, v in m.items()}, f"{time.time()-t0:.0f}s")
del rf

# %% [markdown]
# ### 6.2 Neural baselines and HyperDiag (multiple seeds)

# %%
histories, trained, part_rows = {}, {}, []
for r, Xp in PART.items():
    m, _, _ = rank_metrics(nb.predict_log_proba(sparse.csr_matrix(Xp)), yte)
    part_rows.append(dict(model="Naive Bayes (Bernoulli)", keep_ratio=r, seed=0, **m))
for name, builder in BUILDERS.items():
    for seed in CFG["seeds"]:
        model, hist, tsec = train_neural(builder, seed, Xtr, ytr, Xva, yva, tag=name)
        lo = predict_logits(model, Xte); m, _, _ = rank_metrics(lo, yte)
        results.append(dict(model=name, seed=seed, time_s=tsec, **m)); store[(name, seed)] = lo.numpy()
        for r, Xp in PART.items():
            mp, _, _ = rank_metrics(predict_logits(model, Xp), yte); part_rows.append(dict(model=name, keep_ratio=r, seed=seed, **mp))
        histories[(name, seed)] = hist
        if seed == CFG["seeds"][0]: trained[name] = model
        print(f"==> {name} seed {seed}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" | {tsec:.0f}s")
res_df = pd.DataFrame(results); res_df.to_csv(os.path.join(RES, "all_runs.csv"), index=False)

# %% [markdown]
# ### 6.3 Table 2: Comparative performance (grouped, leakage-free test set; mean ± std over seeds)

# %%
METRICS = ["Top-1", "Top-3", "Top-5", "Top-10", "MRR", "Macro-F1", "Weighted-F1"]
order = ["Naive Bayes (Bernoulli)", "Random Forest"] + list(BUILDERS)
agg = res_df.groupby("model")[METRICS].agg(["mean", "std"]).reindex(order)
def fmt(a):
    out = pd.DataFrame(index=a.index)
    for mtr in METRICS:
        mu, sd = a[(mtr, "mean")] * 100, a[(mtr, "std")].fillna(0) * 100
        out[mtr] = [f"{x:.2f} ± {s:.2f}" for x, s in zip(mu, sd)]
    return out
table2 = fmt(agg)
ceil_test = bayes_ceiling(groups[te_idx], yte)
table2.loc["Bayes-optimal ceiling (test)"] = [f"{ceil_test.get(m, np.nan)*100:.2f}" if m in ceil_test else "—" for m in METRICS]
table2.to_csv(os.path.join(RES, "table2_comparison.csv"))
agg.to_csv(os.path.join(RES, "table2_comparison_raw.csv"))
table2

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
plot_df = res_df.groupby("model")[["Top-1", "Top-5", "MRR", "Macro-F1"]].mean().reindex(order) * 100
plot_df.plot.barh(ax=ax[0], width=0.8, colormap="tab10"); ax[0].set(xlim=(plot_df.values.min() - 5, 100), xlabel="%", title="Test performance (mean over seeds)")
for name in BUILDERS:
    h = histories[(name, CFG["seeds"][0])]; ax[1].plot(h["epoch"], h["val_MRR"], label=name, marker=".")
ax[1].set(xlabel="epoch", ylabel="validation MRR", title="Learning curves (seed 0)"); ax[1].legend(fontsize=8)
savefig("fig_comparison.png")

# %% [markdown]
# ### 6.4 Statistical significance
# McNemar's test (top-1 correctness) and a paired bootstrap of the Top-5/MRR difference between HyperDiag and every baseline (seed 0).

# %%
def mcnemar(c1, c2):
    b = int(((c1 == 1) & (c2 == 0)).sum()); c = int(((c1 == 0) & (c2 == 1)).sum())
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1); return b, c, 1 - chi2.cdf(stat, 1)

def boot_ci(a, b, n=1000, seed=0):
    rs = np.random.RandomState(seed); d = a - b; idx = rs.randint(0, len(d), (n, len(d)))
    bs = d[idx].mean(1); return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

ours = "HyperDiag (ours)"
_, r_ours, _ = rank_metrics(store[(ours, CFG["seeds"][0])], yte)
sig = []
for name in order:
    if name == ours: continue
    _, r_b, _ = rank_metrics(store[(name, 0)], yte)
    b, c, p = mcnemar((r_ours == 1).astype(int), (r_b == 1).astype(int))
    d5, lo5, hi5 = boot_ci((r_ours <= 5).astype(float), (r_b <= 5).astype(float))
    dm, lom, him = boot_ci(1 / r_ours, 1 / r_b)
    sig.append(dict(baseline=name, ours_only_correct=b, baseline_only_correct=c, mcnemar_p=p,
                    dTop5=f"{d5*100:+.2f} [{lo5*100:+.2f}, {hi5*100:+.2f}]", dMRR=f"{dm*100:+.2f} [{lom*100:+.2f}, {him*100:+.2f}]"))
sig_df = pd.DataFrame(sig); sig_df.to_csv(os.path.join(RES, "significance.csv"), index=False); sig_df

# %% [markdown]
# ## 7. Partial-symptom robustness (RQ5)
# Patients often report only part of their symptoms. Each test case keeps a random $\lceil r\cdot k\rceil$ of its $k$ symptoms (at least one).
# The same masks are used for every model (computed during training in §6.2, mean over seeds shown).

# %%

part_df = pd.DataFrame(part_rows); part_df.to_csv(os.path.join(RES, "partial_symptoms.csv"), index=False)
piv = part_df.pivot_table(index="model", columns="keep_ratio", values="Top-5").reindex([o for o in order if o in part_df.model.unique()]) * 100
piv = piv[sorted(piv.columns, reverse=True)]; piv.columns = [f"keep {int(c*100)}%" for c in piv.columns]; piv.round(2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
for mtr, a in zip(["Top-5", "MRR"], ax):
    p = part_df.pivot_table(index="keep_ratio", columns="model", values=mtr) * 100
    for col in p.columns: a.plot(p.index, p[col], marker="o", label=col, lw=2.5 if col == ours else 1.2)
    a.set(xlabel="fraction of symptoms kept", ylabel=f"{mtr} (%)", title=f"Partial-symptom robustness: {mtr}"); a.invert_xaxis()
ax[1].legend(fontsize=8); savefig("fig_partial.png")

# %% [markdown]
# ## 8. Rare vs frequent diseases (RQ6)
# Diseases are grouped by number of training cases. We report per-group recall@1 and recall@5, averaged over the diseases in each group.

# %%
bins = [0, 20, 100, 400, 10**9]; labels_b = ["rare (≤20)", "low (21–100)", "mid (101–400)", "frequent (>400)"]
grp = pd.cut(n_c, bins=bins, labels=labels_b)
rare_rows = []
for name in ["Naive Bayes (Bernoulli)", "MLP", "GAT (pairwise)", "HGNN (hypergraph, no attention)", ours]:
    _, rk, _ = rank_metrics(store[(name, 0)], yte)
    t = pd.DataFrame({"y": yte, "r1": rk <= 1, "r5": rk <= 5})
    pc = t.groupby("y")[["r1", "r5"]].mean(); pc["grp"] = grp[pc.index]
    for gname, gg in pc.groupby("grp", observed=True):
        rare_rows.append(dict(model=name, group=gname, n_diseases=len(gg), recall1=gg.r1.mean(), recall5=gg.r5.mean()))
rare_df = pd.DataFrame(rare_rows); rare_df.to_csv(os.path.join(RES, "rare_diseases.csv"), index=False)
rare_df.pivot_table(index="model", columns="group", values="recall5", observed=True).round(4) * 100

# %% [markdown]
# ## 9. Ablation study (Table 3)
# Each component of HyperDiag is removed in turn and the model is retrained with the same seeds.

# %%
ABL = {
    "Full HyperDiag": {},
    "w/o co-occurrence hyperedges": dict(use_cooc=False),
    "w/o node-level attention": dict(node_attn=False),
    "w/o hyperedge-level attention": dict(edge_attn=False),
    "w/o both attentions": dict(node_attn=False, edge_attn=False),
    "w/o query-aware hyperedge attention": dict(query_edge_attn=False),
    "w/o prototype scoring (MLP head)": dict(proto=False),
    "w/o hyperedge-likelihood evidence": dict(hyper_prior=False),
}
abl_rows = []
for vname, kw in ABL.items():
    for seed in CFG["seeds"]:
        if vname == "Full HyperDiag":
            m, _, _ = rank_metrics(store[(ours, seed)], yte)
            mp = part_df[(part_df.model == ours) & (part_df.seed == seed) & (part_df.keep_ratio == 0.4)].iloc[0].to_dict()
        else:
            mdl, _, _ = train_neural(lambda: HyperDiag(**{**common, **kw}), seed, Xtr, ytr, Xva, yva, verbose=False, tag=vname)
            m, _, _ = rank_metrics(predict_logits(mdl, Xte), yte)
            mp, _, _ = rank_metrics(predict_logits(mdl, PART[0.4]), yte)
        abl_rows.append(dict(variant=vname, seed=seed, **m, **{"Top-5 @40% symptoms": mp["Top-5"]}))
        print(f"{vname:40s} s{seed} top1 {m['Top-1']:.4f} top5 {m['Top-5']:.4f} MRR {m['MRR']:.4f} F1 {m['Macro-F1']:.4f}")
abl_df = pd.DataFrame(abl_rows); abl_df.to_csv(os.path.join(RES, "ablation_runs.csv"), index=False)
AM = ["Top-1", "Top-5", "MRR", "Macro-F1", "Top-5 @40% symptoms"]
a = abl_df.groupby("variant")[AM].agg(["mean", "std"]).reindex(list(ABL))
table3 = pd.DataFrame({mtr: [f"{x*100:.2f} ± {(0 if np.isnan(s) else s)*100:.2f}" for x, s in zip(a[(mtr, "mean")], a[(mtr, "std")])] for mtr in AM}, index=a.index)
table3.to_csv(os.path.join(RES, "table3_ablation.csv")); a.to_csv(os.path.join(RES, "table3_ablation_raw.csv"))
table3

# %% [markdown]
# ## 10. Interpretability analysis (RQ4, RQ7)
# ### 10.1 Case studies: differential diagnosis with symptom-evidence attention
# For a test patient, HyperDiag returns a ranked top-5 differential and the weight $\gamma_v$ of every reported symptom.

# %%
hd = trained[ours].eval()
@torch.no_grad()
def explain(i, k=5):
    M = torch.tensor(Xte[i:i + 1]).to(DEV)
    lo, att = hd(M, return_att=True); p = torch.softmax(lo, 1)[0].cpu().numpy()
    g = att["gamma"][0].cpu().numpy(); on = np.flatnonzero(Xte[i])
    topk = np.argsort(-p)[:k]
    return p, g, on, topk, att

rs = np.random.RandomState(7)
cand = [i for i in rs.permutation(len(yte)) if 4 <= Xte[i].sum() <= 7][:3]
fig, axes = plt.subplots(len(cand), 2, figsize=(15, 3.3 * len(cand)))
case_rows = []
for row, i in zip(axes, cand):
    p, g, on, topk, att = explain(i)
    o = on[np.argsort(-g[on])]
    row[0].barh(sym_names[o][::-1], g[o][::-1], color="#3b6ea8"); row[0].set(title=f"Symptom evidence γ (true: {classes[yte[i]]})", xlabel="attention")
    cols = ["#2a9d55" if t == yte[i] else "#9aa5b1" for t in topk]
    row[1].barh([classes[t] for t in topk][::-1], p[topk][::-1], color=cols[::-1]); row[1].set(title="Top-5 differential (green = true disease)", xlabel="probability")
    case_rows.append(dict(case=int(i), true=classes[yte[i]], symptoms="; ".join(f"{sym_names[v]} ({g[v]:.2f})" for v in o),
                          top5="; ".join(f"{classes[t]} ({p[t]:.2f})" for t in topk)))
plt.tight_layout(); savefig("fig_case_studies.png")
pd.DataFrame(case_rows).to_csv(os.path.join(RES, "case_studies.csv"), index=False)
pd.DataFrame(case_rows)

# %% [markdown]
# ### 10.2 Node-level attention inside disease hyperedges
# For a disease hyperedge, the layer-1 node-level attention $\alpha_{v,d}$ (mean over heads) shows which member symptoms the model treats as most characteristic.
# We compare it with the empirical prevalence $P(s\mid d)$ and with the specificity $P(d\mid s)$.

# %%
with torch.no_grad():
    _, _, atts = hd.encode()
alpha1 = atts[0][0].mean(-1).cpu().numpy()[:, :C]           # S x C (disease hyperedges)
p_ds = (ST["p_sd"] * n_c[:, None]) / np.maximum((ST["p_sd"] * n_c[:, None]).sum(0, keepdims=True), 1e-9)  # P(d|s), C x S
rho_prev, rho_spec = [], []
for d in range(C):
    mem = np.flatnonzero(ST["Hd"][d])
    if len(mem) >= 4:
        rho_prev.append(spearmanr(alpha1[mem, d], ST["p_sd"][d, mem]).correlation)
        rho_spec.append(spearmanr(alpha1[mem, d], p_ds[d, mem]).correlation)
att_corr = pd.DataFrame({"comparison": ["α vs prevalence P(s|d)", "α vs specificity P(d|s)"],
                         "mean Spearman ρ": [np.nanmean(rho_prev), np.nanmean(rho_spec)],
                         "share of diseases with ρ>0": [np.nanmean(np.array(rho_prev) > 0), np.nanmean(np.array(rho_spec) > 0)],
                         "#diseases": [len(rho_prev)] * 2}).round(3)
att_corr.to_csv(os.path.join(RES, "attention_alignment.csv"), index=False)
show(att_corr)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for a_, d in zip(axes, np.argsort(-n_c)[[0, 3, 6]]):
    mem = np.flatnonzero(ST["Hd"][d]); mem = mem[np.argsort(-alpha1[mem, d])][:10]
    xx = np.arange(len(mem)); w = 0.4
    a_.barh(xx - w / 2, alpha1[mem, d] / alpha1[mem, d].max(), w, label="node attention α (scaled)", color="#3b6ea8")
    a_.barh(xx + w / 2, ST["p_sd"][d, mem] / ST["p_sd"][d, mem].max(), w, label="P(s|d) (scaled)", color="#d07a3a")
    a_.set_yticks(xx); a_.set_yticklabels(sym_names[mem], fontsize=8); a_.invert_yaxis(); a_.set_title(classes[d])
axes[0].legend(fontsize=8); plt.tight_layout(); savefig("fig_node_attention.png")

# %% [markdown]
# ### 10.3 Faithfulness of the explanations (deletion test)
# If $\gamma$ is a faithful explanation, deleting the *most*-attended symptom should hurt the prediction more than deleting a random one,
# and much more than deleting the *least*-attended one. We use test cases with at least 3 symptoms.

# %%
@torch.no_grad()
def deletion_test(model, Xm, ym, n=3000, seed=0):
    rs = np.random.RandomState(seed); idx = [i for i in rs.permutation(len(ym)) if Xm[i].sum() >= 3][:n]
    Xs = Xm[idx]; ys = ym[idx]; M = torch.tensor(Xs).to(DEV)
    lo, att = model(M, return_att=True); g = att["gamma"].cpu().numpy()
    base_p = torch.softmax(lo, 1).cpu().numpy()[np.arange(len(ys)), ys]
    out = {}
    for mode in ["most attended", "random", "least attended"]:
        Xd = Xs.copy()
        for j in range(len(ys)):
            on = np.flatnonzero(Xs[j])
            v = on[np.argmax(g[j, on])] if mode == "most attended" else on[np.argmin(g[j, on])] if mode == "least attended" else rs.choice(on)
            Xd[j, v] = 0
        lo2 = predict_logits(model, Xd); m, rk, _ = rank_metrics(lo2, ys)
        p2 = torch.softmax(lo2, 1).numpy()[np.arange(len(ys)), ys]
        out[mode] = dict(mean_drop_true_prob=float((base_p - p2).mean()), Top1_after=m["Top-1"], Top5_after=m["Top-5"])
    m0, _, _ = rank_metrics(lo.cpu(), ys); out["no deletion"] = dict(mean_drop_true_prob=0.0, Top1_after=m0["Top-1"], Top5_after=m0["Top-5"])
    return pd.DataFrame(out).T

faith = deletion_test(hd, Xte, yte); faith.to_csv(os.path.join(RES, "faithfulness.csv")); faith.round(4)

# %% [markdown]
# ## 11. Summary of results
# The cell below gathers the main findings into `results/summary.json`, which is used to fill the thesis draft.

# %%
summary = dict(
    config=CFG, audit=audit, bayes_ceiling=ceiling_df.to_dict(), test_ceiling=ceil_test,
    split=dict(train=len(tr_idx), val=len(va_idx), test=len(te_idx), symptoms=S, diseases=C),
    hypergraph=hstats.to_dict(orient="records"),
    leakage=leak_df.to_dict(orient="records"),
    table2={m_: {f"{k1} {k2}": v for (k1, k2), v in row.items()} for m_, row in agg.round(5).iterrows()},
    table3={m_: {f"{k1} {k2}": v for (k1, k2), v in row.items()} for m_, row in a.round(5).iterrows()},
    partial=piv.round(3).reset_index().to_dict(orient="records"),
    rare=rare_df.round(4).to_dict(orient="records"),
    significance=sig_df.to_dict(orient="records"),
    attention_alignment=att_corr.to_dict(orient="records"),
    faithfulness=faith.round(4).reset_index().to_dict(orient="records"),
    case_studies=case_rows,
)
with open(os.path.join(RES, "summary.json"), "w") as fh: json.dump(summary, fh, indent=1, default=str)
best_base = agg.drop(index=ours)[("MRR", "mean")].idxmax()
print(f"HyperDiag:      Top-1 {agg.loc[ours, ('Top-1','mean')]*100:.2f} | Top-5 {agg.loc[ours, ('Top-5','mean')]*100:.2f} | MRR {agg.loc[ours, ('MRR','mean')]*100:.2f} | Macro-F1 {agg.loc[ours, ('Macro-F1','mean')]*100:.2f}")
print(f"Best baseline ({best_base}): Top-1 {agg.loc[best_base, ('Top-1','mean')]*100:.2f} | Top-5 {agg.loc[best_base, ('Top-5','mean')]*100:.2f} | MRR {agg.loc[best_base, ('MRR','mean')]*100:.2f} | Macro-F1 {agg.loc[best_base, ('Macro-F1','mean')]*100:.2f}")
print(f"Bayes ceiling on test: Top-1 {ceil_test['Top-1']*100:.2f} | Top-5 {ceil_test['Top-5']*100:.2f}")
