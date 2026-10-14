# %% [markdown]
# # GiftLane: Customers Like You  (ML1-ML7)
# Analysis notebook. Run `python build_matrix.py` first (it rebuilds the matrix from the raw file).
# All customers are shown only by anonymised row index; no customer IDs appear in any output (R4).

# %%
import json, os, threading, time
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
np.random.seed(SEED)
PROC, OUT = "data/processed", "outputs"
os.makedirs(f"{OUT}/figures", exist_ok=True)
K_GRID = [10, 20, 50, 100, 200]
RESULTS = {}

A = sp.load_npz(f"{PROC}/A_log.npz").tocsr()
products = pd.read_csv(f"{PROC}/products.csv", dtype={"stockcode": str})
customers = pd.read_csv(f"{PROC}/customers.csv")            # private: IDs never printed
held = pd.read_csv(f"{PROC}/heldout.csv")
held_sets = held.groupby("customer_idx")["product_idx"].apply(set).to_dict()
test_cust = np.array(sorted(held_sets))
print("matrix", A.shape, "nnz", A.nnz, "| test customers", len(test_cust), "| held-out pairs", len(held))

# %% [markdown]
# ## ML1  Decision note (written before any model was fitted)
# **Main model: Truncated SVD.** Reasons: (1) the purchase matrix is ~98% zeros and SVD works on the sparse
# matrix directly, while PCA must centre it (implicitly) and then describes *deviations from the average
# customer*; (2) we need product factors for "customers who bought this also bought" and SVD hands us
# customer and product factors in one fit; (3) PCA centring would turn "did not buy" into a small
# negative signal, which is not what a recommender wants. PCA is run second (ML5) as the comparison.
#
# **How k is chosen:** on a *validation* split made inside the training period (never the test period, R2),
# plus the reconstruction-error curve. **Evaluation:** hit rate at 10 on products the customer had never bought
# before, against the best-seller baseline (R3).

# %%
# ---------- helpers
class PeakMem:
    """Sample total process RSS during fit, including the already loaded data."""
    def __enter__(self):
        self.p = psutil.Process(); self.base = self.p.memory_info().rss; self.peak = self.base; self.go = True
        def loop():
            while self.go:
                self.peak = max(self.peak, self.p.memory_info().rss); time.sleep(0.01)
        self.t = threading.Thread(target=loop); self.t.start(); return self
    def __exit__(self, *a):
        self.go = False; self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)
        self.mb = self.peak / 1e6

def top_n(scores, n=10):
    idx = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def pop_prior(nb):
    p = np.log1p(nb); return p / p.max()

def model_scores(A_rows, P, pop=None, alpha=0.0):
    """Score every product for each customer: similarity to what they bought (ML3).
    alpha > 0 adds a popularity prior (hybrid): score = sim / max|sim| + alpha * popularity."""
    s = np.asarray((A_rows @ P) @ P.T)
    if alpha:
        s = s / (np.abs(s).max(1, keepdims=True) + 1e-9) + alpha * pop
    s[A_rows.toarray() > 0] = -np.inf          # never recommend what they already own
    return s

def bestseller_scores(A_rows, n_buyers):
    s = np.tile(n_buyers.astype(float), (A_rows.shape[0], 1))
    s[A_rows.toarray() > 0] = -np.inf
    return s

def hit_rate(scores, cust_idx, hs, n=10):
    top = top_n(scores, n)
    return float(np.mean([len(hs[c] & set(t)) > 0 for c, t in zip(cust_idx, top)]))

def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)

def product_vectors(model):
    return normalize(model.components_.T)

def fit_model(kind, M, k):
    if kind == "svd":
        return TruncatedSVD(n_components=k, random_state=SEED).fit(M)
    return PCA(n_components=k, svd_solver="arpack", random_state=SEED).fit(M)

def split_validation(train_csv, months=2, min_c=10):
    """Inner split of the TRAINING period only: fit on the earlier part, validate on the last 2 months of it."""
    tr = pd.read_csv(train_csv, parse_dates=["InvoiceDate"], dtype={"StockCode": str})
    cut = tr["InvoiceDate"].max().normalize() - pd.DateOffset(months=months)
    a, b = tr[tr["InvoiceDate"] < cut], tr[tr["InvoiceDate"] >= cut]
    n_c = a.groupby("StockCode")["Customer ID"].nunique()
    a = a[a["StockCode"].isin(n_c[n_c >= min_c].index)]
    agg = a.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()
    cm = {c: i for i, c in enumerate(np.sort(agg["Customer ID"].unique()))}
    pm = {p: j for j, p in enumerate(sorted(agg["StockCode"].unique()))}
    r, c = agg["Customer ID"].map(cm).to_numpy(), agg["StockCode"].map(pm).to_numpy()
    Av = sp.csr_matrix((np.log1p(agg["Quantity"].to_numpy(float)), (r, c)), shape=(len(cm), len(pm)))
    bought = set(zip(r.tolist(), c.tolist()))
    b = b[b["Customer ID"].isin(cm) & b["StockCode"].isin(pm)][["Customer ID", "StockCode"]].drop_duplicates()
    pairs = [(cm[x], pm[y]) for x, y in zip(b["Customer ID"], b["StockCode"]) if (cm[x], pm[y]) not in bought]
    hs = {}
    for ci, pj in pairs: hs.setdefault(ci, set()).add(pj)
    nb = np.asarray((Av > 0).sum(0)).ravel()
    return Av, hs, nb

# %% [markdown]
# ## ML2  Choosing k
# Reconstruction error of Truncated SVD (what compression loses) and PCA explained variance on the real
# training matrix; hit rate at 10 on the **validation** split (inside the training period).

# %%
Av, hs_v, nb_v = split_validation(f"{PROC}/train.csv.gz")
vc = np.array(sorted(hs_v))
print("validation matrix", Av.shape, "| validation customers", len(vc))
fro2 = float(A.multiply(A).sum())
n_buyers = np.asarray((A > 0).sum(0)).ravel()
ALPHAS = [0, 0.5, 1, 2, 3, 5]          # popularity-prior weights tried (0 = pure SVD similarity)
pop_v = pop_prior(nb_v)
rows, grid = [], []
for k in K_GRID:
    svd = fit_model("svd", A, k)
    err = float(np.sqrt(max(1 - (svd.singular_values_ ** 2).sum() / fro2, 0)))
    pca = fit_model("pca", A, k)
    # validation hit rates (fit on the earlier part of training only)
    Ps = product_vectors(fit_model("svd", Av, k)); Pp = product_vectors(fit_model("pca", Av, k))
    hs_ = {a: hit_rate(model_scores(Av[vc], Ps, pop_v, a), vc, hs_v) for a in ALPHAS}
    hp_ = {a: hit_rate(model_scores(Av[vc], Pp, pop_v, a), vc, hs_v) for a in ALPHAS}
    for a in ALPHAS: grid.append(dict(k=k, alpha=a, svd_val_hit10=hs_[a], pca_val_hit10=hp_[a]))
    rows.append(dict(k=k, svd_rel_error=err, pca_explained_var=float(pca.explained_variance_ratio_.sum()),
                     svd_val_hit10_pure=hs_[0], pca_val_hit10_pure=hp_[0],
                     svd_val_hit10_hybrid=max(hs_.values()), pca_val_hit10_hybrid=max(hp_.values())))
    print(rows[-1])
curve = pd.DataFrame(rows)
curve["val_baseline_hit10"] = hit_rate(bestseller_scores(Av[vc], nb_v), vc, hs_v)
curve.to_csv(f"{OUT}/k_curve.csv", index=False)
pd.DataFrame(grid).to_csv(f"{OUT}/k_alpha_grid_validation.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(curve.k, curve.svd_rel_error, "o-", color="#1f6fb2")
ax[0].set(xlabel="k (components)", ylabel="relative reconstruction error", title="What compression loses (SVD)")
ax[1].plot(curve.k, curve.svd_val_hit10_hybrid, "o-", color="#1f6fb2", label="SVD + popularity prior")
ax[1].plot(curve.k, curve.svd_val_hit10_pure, "o--", color="#1f6fb2", alpha=.5, label="SVD alone")
ax[1].plot(curve.k, curve.pca_val_hit10_hybrid, "s-", color="#d9822b", label="PCA + popularity prior")
ax[1].axhline(curve.val_baseline_hit10.iloc[0], color="grey", ls=":", label="best sellers")
ax[1].set(xlabel="k (components)", ylabel="hit rate @10 (validation)", title="Suggestion quality vs k"); ax[1].legend(fontsize=7)
for a_ in ax: a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/k_curve.png", dpi=150); plt.close(fig)

best = curve.svd_val_hit10_hybrid.max()
K = int(curve[curve.svd_val_hit10_hybrid >= 0.97 * best].k.min())   # smallest k within 3% of the best validation hit rate
g_ = pd.DataFrame(grid); g_ = g_[g_.k == K]
ALPHA = float(g_.loc[g_.svd_val_hit10.idxmax(), "alpha"])
RESULTS.update(k_chosen=K, alpha_chosen=ALPHA,
               k_rule="smallest k whose SVD validation hit rate is within 3% of the best; alpha = best on validation at that k")
print("chosen k =", K, " alpha =", ALPHA)

# %% [markdown]
# ## ML3 / ML4 / ML5  Recommendations, evaluation on the untouched test period, PCA vs SVD
# Timed model stage = fit + top-10 similarities for every product + scores for eligible test customers.
# Excel loading, cleaning, matrix building and saving are excluded. Factor sizes below are calculated
# float64 storage estimates, not measured model files. The sparse input can be smaller than the factors.

# %%
def full_job(kind, k):
    with PeakMem() as pm_:
        t0 = time.time(); model = fit_model(kind, A, k); fit_s = time.time() - t0
    t1 = time.time()
    P = product_vectors(model)
    S = P @ P.T; np.fill_diagonal(S, -np.inf)
    similar_products = top_n(S, 10)                      # "customers who bought this also bought" table
    sc = model_scores(A[test_cust], P, pop_t, ALPHA)
    top = top_n(sc, 10)
    score_s = time.time() - t1
    hit = float(np.mean([len(held_sets[c] & set(t)) > 0 for c, t in zip(test_cust, top)]))
    hit_pure = hit_rate(model_scores(A[test_cust], P), test_cust, held_sets)
    factors_mb = ((A.shape[0] + A.shape[1]) * k * 8 + (A.shape[1] * 8 if kind == "pca" else 0)) / 1e6
    return dict(model=model, P=P, similar=similar_products, fit_s=fit_s, score_s=score_s, peak_mb=pm_.mb,
                factors_mb=factors_mb, hit=hit, hit_pure=hit_pure, top=top)

pop_t = pop_prior(n_buyers)
res = {kd: full_job(kd, K) for kd in ("svd", "pca")}
n_t = len(test_cust)
base_hit = hit_rate(bestseller_scores(A[test_cust], n_buyers), test_cust, held_sets)
# uncompressed reference: item-item cosine on the full sparse matrix (what compression gives up)
t0 = time.time(); Pfull = normalize(A.T.tocsr()); Sfull = (Pfull @ Pfull.T).toarray(); np.fill_diagonal(Sfull, 0)
sc_full = np.asarray(A[test_cust] @ Sfull)
full_hit_pure = hit_rate(np.where(A[test_cust].toarray() > 0, -np.inf, sc_full), test_cust, held_sets)
sc_full = sc_full / (np.abs(sc_full).max(1, keepdims=True) + 1e-9) + ALPHA * pop_t
sc_full[A[test_cust].toarray() > 0] = -np.inf
full_hit = hit_rate(sc_full, test_cust, held_sets); full_s = time.time() - t0

svd_m, pca_m = res["svd"]["model"], res["pca"]["model"]
svd_err = float(np.sqrt(max(1 - (svd_m.singular_values_ ** 2).sum() / fro2, 0)))
svd_ev = float(svd_m.explained_variance_ratio_.sum())
pca_ev = float(pca_m.explained_variance_ratio_.sum())

# what is component 1?  correlate customer scores with simple activity measures
n_items = np.asarray((A > 0).sum(1)).ravel(); volume = np.asarray(A.sum(1)).ravel()
def comp1_info(kind, m):
    Z = m.transform(A)[:, 0]
    top_p = products.assign(w=m.components_[0]).sort_values("w", key=abs, ascending=False).head(5).description.tolist()
    return dict(corr_with_n_products=float(np.corrcoef(Z, n_items)[0, 1]),
                corr_with_volume=float(np.corrcoef(Z, volume)[0, 1]), top_products=top_p,
                share_same_sign=float(max(np.mean(m.components_[0] > 0), np.mean(m.components_[0] < 0))))
c1 = {"svd": comp1_info("svd", svd_m), "pca": comp1_info("pca", pca_m)}
print(json.dumps(c1, indent=1))

table = pd.DataFrame({
    "Measure": ["k used", "Fit time (s)", "Peak memory during fit (MB, total process RSS, sampled)", "Size of stored factors (MB, calculated float64)",
                "Variance explained / relative reconstruction error",
                "Hit rate at 10, factors alone", f"Hit rate at 10, factors + popularity prior (alpha={ALPHA:g})",
                "Best-seller baseline hit rate at 10", "Time to score products + eligible test customers (s)",
                "What component 1 represents, in words"],
    "PCA": [K, round(res["pca"]["fit_s"], 2), round(res["pca"]["peak_mb"], 1), round(res["pca"]["factors_mb"], 2),
            f"{pca_ev:.1%} of variance (centred)", f"{res['pca']['hit_pure']:.1%}", f"{res['pca']['hit']:.1%}",
            f"{base_hit:.1%}", round(res["pca"]["score_s"], 2),
            f"Overall log purchase activity relative to the average customer; r={c1['pca']['corr_with_volume']:.2f} with row log-value sum"],
    "Truncated SVD": [K, round(res["svd"]["fit_s"], 2), round(res["svd"]["peak_mb"], 1), round(res["svd"]["factors_mb"], 2),
                      f"error {svd_err:.1%} ({svd_ev:.1%} of variance, uncentred)", f"{res['svd']['hit_pure']:.1%}",
                      f"{res['svd']['hit']:.1%}", f"{base_hit:.1%}", round(res["svd"]["score_s"], 2),
                      f"Common overall log purchase activity in the uncentred matrix; r={c1['svd']['corr_with_volume']:.2f} with row log-value sum"],
})
table.to_csv(f"{OUT}/comparison_table.csv", index=False)
print(table.to_string(index=False))
RESULTS.update(dict(
    test_customers=int(n_t), baseline_hit10=base_hit, baseline_ci=ci95(base_hit, n_t),
    svd_hit10=res["svd"]["hit"], svd_ci=ci95(res["svd"]["hit"], n_t), svd_hit10_pure=res["svd"]["hit_pure"],
    pca_hit10=res["pca"]["hit"], pca_ci=ci95(res["pca"]["hit"], n_t), pca_hit10_pure=res["pca"]["hit_pure"],
    uncompressed_item_cosine_hit10=full_hit, uncompressed_item_cosine_hit10_pure=full_hit_pure, uncompressed_time_s=full_s,
    svd_fit_s=res["svd"]["fit_s"], svd_score_s=res["svd"]["score_s"], svd_peak_mb=res["svd"]["peak_mb"],
    pca_fit_s=res["pca"]["fit_s"], pca_score_s=res["pca"]["score_s"], pca_peak_mb=res["pca"]["peak_mb"],
    svd_factors_mb=res["svd"]["factors_mb"], pca_factors_mb=res["pca"]["factors_mb"],
    dense_MB=A.shape[0] * A.shape[1] * 8 / 1e6,
    sparse_MB=(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6,
    timing_scope=f"Fit, all product top-10 similarities and scoring {n_t} eligible test customers; excludes preprocessing and saving",
    factor_storage_scope="Calculated float64 customer and product factor storage; factors are not saved",
    baseline_definition="Rank products by distinct training customers; exclude products already bought",
    svd_rel_error=svd_err, svd_explained_var=svd_ev, pca_explained_var=pca_ev,
    comp1=c1))

# example "also bought" box (product names only)
ex = []
for j in np.argsort(-n_buyers)[:3]:
    ex.append({"product": products.description[j], "also_bought": [products.description[i] for i in res["svd"]["similar"][j][:5]]})
RESULTS["also_bought_examples"] = ex
for e in ex: print(e)

# %% [markdown]
# ## ML7  Sensitivity to the purchase value (count, 1/0, log)  [stretch]

# %%
sens = []
for name in ("log", "binary", "count"):
    M = sp.load_npz(f"{PROC}/A_{name}.npz").tocsr()
    m = fit_model("svd", M, K); P = product_vectors(m)
    sens.append(dict(purchase_value=name,
                     svd_hit10_pure=hit_rate(model_scores(M[test_cust], P), test_cust, held_sets),
                     svd_hit10_hybrid=hit_rate(model_scores(M[test_cust], P, pop_t, ALPHA), test_cust, held_sets)))
sens = pd.DataFrame(sens); sens.to_csv(f"{OUT}/sensitivity_value.csv", index=False); print(sens)
RESULTS["sensitivity"] = sens.to_dict("records")

# %% [markdown]
# ## ML6  Customer map
# Following the brief, first project customers onto two components, then cluster those same 2-D
# coordinates with k-means (3 to 6 groups, selected by silhouette). Before projection, scale each
# customer's SVD factors to unit length to reduce the effect of purchasing volume. Segment names
# describe the resulting top products, not the previous clustering in 100 dimensions. Points are anonymous.

# %%
Z = normalize(res["svd"]["model"].transform(A))
XY = PCA(2, random_state=SEED).fit_transform(Z)
sil = {}
for g in range(3, 7):
    lab = KMeans(g, n_init=10, random_state=SEED).fit_predict(XY)
    sil[g] = float(silhouette_score(XY, lab, sample_size=3000, random_state=SEED))
G = max(sil, key=sil.get); print("silhouette", sil, "-> groups", G)
labels = KMeans(G, n_init=10, random_state=SEED).fit_predict(XY)

names_file = f"{OUT}/segment_names.json"
names = json.load(open(names_file, encoding="utf-8-sig")) if os.path.exists(names_file) else {}
Ab = (A > 0).tocsc()
share_all = np.asarray(Ab.mean(0)).ravel()
prof = []
for g in range(G):
    m_ = labels == g
    sh = np.asarray(Ab[m_].mean(0)).ravel()
    lift = sh / np.maximum(share_all, 1e-9)
    ok = np.where(sh >= 0.15)[0]
    top = ok[np.argsort(-lift[ok])][:6] if len(ok) >= 3 else np.argsort(-sh)[:6]
    ctry = customers.country[m_].value_counts(normalize=True).head(3)
    prof.append(dict(segment=g, name=names.get(str(g), f"segment {g}"), customers=int(m_.sum()),
                     share_pct=round(100 * m_.mean(), 1), avg_products=round(float(n_items[m_].mean()), 1),
                     avg_units=round(float(np.expm1(A[m_].data).sum() / m_.sum()), 0),
                     top_products="; ".join(products.description[top].astype(str)),
                     top_countries=", ".join(f"{c} {v:.0%}" for c, v in ctry.items())))
prof = pd.DataFrame(prof); prof.to_csv(f"{OUT}/segment_profile.csv", index=False)
print(prof.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
cols_ = ["#1f6fb2", "#d9822b", "#2e9e6b", "#8a5fbf", "#c4443d", "#8c8c3a"]
for g in range(G):
    m_ = labels == g
    ax.scatter(XY[m_, 0], XY[m_, 1], s=7, alpha=.55, color=cols_[g],
               label=f"{prof.name[g]} ({prof.share_pct[g]:.0f}%)")
ax.set(xlabel="component 1 of normalized customer factors", ylabel="component 2 of normalized customer factors",
       title=f"GiftLane customer map: {G} groups clustered in these 2 components")
ax.legend(markerscale=3, fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/customer_map.png", dpi=150); plt.close(fig)
RESULTS["segments"] = prof.to_dict("records"); RESULTS["silhouette"] = sil
RESULTS["clustering_space"] = "Two PCA components of unit-length SVD customer factors; k-means fitted on plotted coordinates"
json.dump(RESULTS, open(f"{OUT}/results.json", "w"), indent=2, default=float)
print("saved results.json")
