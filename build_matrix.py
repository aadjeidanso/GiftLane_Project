"""Rebuild everything the model needs from the raw Excel file (DE1-DE6, R6).

Usage:  python build_matrix.py

Outputs
  outputs/data_profile.csv      DE1  profile of the raw file
  outputs/cleaning_log.csv      DE2  rows removed by each rule
  data/processed/train.csv.gz   DE3  cleaned training lines (before the cutoff)
  data/processed/test.csv.gz    DE3  cleaned test-period lines
  data/processed/heldout.csv    DE3  (customer_idx, product_idx) never bought in training
  data/processed/A_*.npz        DE4  sparse customer x product matrices (log / count / binary)
  data/processed/customers.csv  DE4  index map (private: contains customer IDs, never leaves the team)
  data/processed/products.csv   DE4  index map with descriptions
  outputs/matrix_report.json    DE5  shape, nnz, sparsity, dense vs sparse memory
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 42
RAW = "data/raw/online_retail_II.xlsx"
CACHE = "data/interim/raw.pkl"
PROC = "data/processed"
OUT = "outputs"
MIN_CUSTOMERS_PER_PRODUCT = 10   # product filter (R1: see decision_log.md)
TEST_MONTHS = 2                  # last two months are the test period

np.random.seed(SEED)
for d in (PROC, OUT, "data/interim"):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- DE1: load + profile
t0 = time.time()
if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
else:
    sheets = pd.read_excel(RAW, sheet_name=None)
    for name, s in sheets.items():
        print(f"sheet {name!r}: {len(s):,} rows, columns {s.columns.tolist()}")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_pickle(CACHE)
print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")

df = df.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"})
profile = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "rows": len(df),
    "missing": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2),
    "unique": df.nunique(),
})
profile.index.name = "column"
profile.to_csv(f"{OUT}/data_profile.csv")
print(profile)

# ---------------------------------------------------------------- DE2: clean (each rule logged)
df["Invoice"] = df["Invoice"].astype(str)
df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
raw_rows = len(df)

# a real product code is 5 digits plus up to 2 letters (variants); POST, M, D, BANK CHARGES,
# DOT, TEST001, gift_0001_20 ... are not products
PRODUCT_RE = re.compile(r"^\d{5}[A-Z]{0,2}$")
rules = [
    ("cancellation invoice (starts with C)", lambda d: d["Invoice"].str.startswith("C")),
    ("quantity <= 0", lambda d: d["Quantity"] <= 0),
    ("price <= 0", lambda d: d["Price"] <= 0),
    ("non-product stock code", lambda d: ~d["StockCode"].str.match(PRODUCT_RE)),
    ("missing customer ID", lambda d: d["Customer ID"].isna()),
]
log = []
for name, rule in rules:
    mask = rule(df)
    log.append({"rule": name, "flagged_alone": int(mask.sum()),
                "removed_in_sequence": int(mask.sum()), "rows_after": 0})
    df = df[~mask]
    log[-1]["rows_after"] = len(df)
# 'removed_in_sequence' is what the rule removed after earlier rules had already run
log.insert(0, {"rule": "raw file", "flagged_alone": 0, "removed_in_sequence": 0, "rows_after": raw_rows})
# recompute flagged_alone on the raw rows so the table shows both views
raw_df = pd.read_pickle(CACHE).rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"})
raw_df["Invoice"] = raw_df["Invoice"].astype(str)
raw_df["StockCode"] = raw_df["StockCode"].astype(str).str.strip().str.upper()
for entry, (_, rule) in zip(log[1:], rules):
    entry["flagged_alone"] = int(rule(raw_df).sum())
del raw_df
cleaning_log = pd.DataFrame(log)
cleaning_log.to_csv(f"{OUT}/cleaning_log.csv", index=False)
print(cleaning_log.to_string(index=False))

df["Customer ID"] = df["Customer ID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df.reset_index(drop=True)

# product description = most frequent non-null description of the code
desc = (df.dropna(subset=["Description"]).groupby("StockCode")["Description"]
        .agg(lambda s: s.str.strip().mode().iloc[0]))

# ---------------------------------------------------------------- DE3: time split
cutoff = df["InvoiceDate"].max().normalize() - pd.DateOffset(months=TEST_MONTHS)
train = df[df["InvoiceDate"] < cutoff].copy()
test = df[df["InvoiceDate"] >= cutoff].copy()
print(f"cutoff {cutoff.date()}  train {len(train):,} rows  test {len(test):,} rows")
train.to_csv(f"{PROC}/train.csv.gz", index=False)
test.to_csv(f"{PROC}/test.csv.gz", index=False)

# ---------------------------------------------------------------- DE4: matrix (training rows only)
n_cust = train.groupby("StockCode")["Customer ID"].nunique()
keep = n_cust[n_cust >= MIN_CUSTOMERS_PER_PRODUCT].index
train_k = train[train["StockCode"].isin(keep)]
# one cell per (customer, product): total units bought over the training period
agg = train_k.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()

cust_ids = np.sort(agg["Customer ID"].unique())
prod_codes = np.array(sorted(agg["StockCode"].unique()))
cust_map = {c: i for i, c in enumerate(cust_ids)}
prod_map = {p: j for j, p in enumerate(prod_codes)}
rows = agg["Customer ID"].map(cust_map).to_numpy()
cols = agg["StockCode"].map(prod_map).to_numpy()
qty = agg["Quantity"].to_numpy(dtype=float)
shape = (len(cust_ids), len(prod_codes))

values = {"log": np.log1p(qty), "count": qty, "binary": np.ones_like(qty)}
for name, v in values.items():
    sp.save_npz(f"{PROC}/A_{name}.npz", sp.csr_matrix((v, (rows, cols)), shape=shape))

country = (train.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iloc[0]))
pd.DataFrame({"customer_idx": np.arange(len(cust_ids)), "customer_id": cust_ids,
              "country": country.reindex(cust_ids).to_numpy()}).to_csv(f"{PROC}/customers.csv", index=False)
pd.DataFrame({"product_idx": np.arange(len(prod_codes)), "stockcode": prod_codes,
              "description": desc.reindex(prod_codes).to_numpy(),
              "n_customers": n_cust.reindex(prod_codes).to_numpy()}).to_csv(f"{PROC}/products.csv", index=False)

# held-out purchases: test products the customer never bought in training (R2)
bought_train = set(zip(rows.tolist(), cols.tolist()))
held = test[test["Customer ID"].isin(cust_map) & test["StockCode"].isin(prod_map)]
held = held[["Customer ID", "StockCode"]].drop_duplicates()
held = pd.DataFrame({"customer_idx": held["Customer ID"].map(cust_map).to_numpy(),
                     "product_idx": held["StockCode"].map(prod_map).to_numpy()})
held = held[[(c, p) not in bought_train for c, p in zip(held["customer_idx"], held["product_idx"])]]
held.to_csv(f"{PROC}/heldout.csv", index=False)

# ---------------------------------------------------------------- DE5: matrix report
A = sp.load_npz(f"{PROC}/A_log.npz")
dense_mb = shape[0] * shape[1] * 8 / 1e6
sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
report = {
    "raw_rows": raw_rows, "rows_after_cleaning": len(df), "cutoff": str(cutoff.date()),
    "train_rows": len(train), "test_rows": len(test),
    "customers_in_train_matrix": shape[0], "products_total_in_train": int(n_cust.size),
    "products_kept_(>=10 customers)": shape[1],
    "shape": list(shape), "nnz": int(A.nnz),
    "sparsity_pct": round(100 * (1 - A.nnz / (shape[0] * shape[1])), 2),
    "dense_MB": round(dense_mb, 1), "sparse_MB": round(sparse_mb, 2),
    "test_customers_with_heldout": int(held["customer_idx"].nunique()),
    "heldout_pairs": len(held),
}
json.dump(report, open(f"{OUT}/matrix_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
print(f"done in {time.time() - t0:.0f}s")
