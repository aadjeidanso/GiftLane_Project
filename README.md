# GiftLane: Customers Like You (PCA vs SVD)

Emmanuel Adjei Danso - Solo project (Data Engineer + ML Engineer). Data: UCI Online Retail II, **1,067,371 rows** across 2 sheets (525,461 + 541,910), 1 Dec 2009 to 9 Dec 2011.

## Success targets (J1, written Day 1)

1. Compressed data much smaller than the full matrix (target: > 10x smaller).
2. Nightly job fast (target: under 1 minute).
3. Recommendations beat the best-seller baseline on held-out purchases (hit rate @10).
4. Customer map with 3 to 6 groups Marketing can name, no customer IDs.

## Results against targets

| Target        | Result                                                                                                                                   |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Size          | Calculated float64 storage: 161.3 MB dense -> 7.33 MB factors (k = 100), 22x smaller than dense. Sparse input: 4.99 MB, already smaller than factors |
| Speed         | Model stage: fit 0.41 s + product similarities and scoring 1,913 test customers 0.33 s; excludes loading, cleaning, matrix construction and saving |
| Beat baseline | SVD + popularity prior 36.6% vs best sellers 27.0% (1,913 test customers, +/-2 pts). SVD factors alone: 22.5% (loses, reported honestly) |
| Customer map  | 3 named groups, clustered on the two displayed components; `outputs/figures/customer_map.png` |

## How to run

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# put online_retail_II.xlsx in data/raw/
python build_matrix.py        # DE1-DE6: profile, cleaning log, split, sparse matrices (R6)
python make_notebook.py       # runs analysis.py as analysis.ipynb top to bottom (ML1-ML7)
python make_deliverables.py   # memo.pdf and slides.pptx from outputs/results.json
```

Seeds are fixed (42). `data/` is git-ignored (raw data and private customer-ID index maps).

Factor sizes are calculated storage estimates; the pipeline does not save factor model files. The best-seller baseline ranks products by distinct training customers and excludes previous purchases. The 36.6% hit rate evaluates customer scoring with a popularity prior; product-page suggestions use factor similarity alone and are not directly evaluated by this customer metric. Runtime varies by machine and run. The comparison table reports sampled total process memory during fitting, including already loaded data and earlier allocations.

For ML6, unit-length SVD customer factors are projected onto two PCA components, then k-means is fitted on those same two coordinates. The silhouette score selects 3 groups from 3 to 6: Mixed gifts and baking shops (46%), Heart and home-decor shops (30%), and Jumbo-bag buyers (24%). This replaces the earlier five-group clustering in 100 dimensions.

## Files

- `build_matrix.py` data engineering; `analysis.py`/`analysis.ipynb` modelling; `decision_log.md` every decision with reason (R1)
- `outputs/` cleaning log, data profile, matrix report, k curve, PCA vs SVD table (`comparison_table.csv`), sensitivity, segment profile, figures
- `memo.pdf` (J2), `slides.pptx` (J3), `contribution_log.md` (C2), `reflection.md`

## Roles

One person held both roles: Data Engineer (DE1-DE6) and ML Engineer (ML1-ML7). Stretch tasks DE6 and ML7 were done too.
