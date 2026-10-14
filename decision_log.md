# Decision log (R1)

| # | Decision | Reason |
|---|---|---|
| 1 | Use both sheets (1,067,371 rows) | Two years of history allow a time-based split with a full year of training before the test period |
| 2 | Remove cancellations (invoice starts with "C"), quantity <= 0, price <= 0 | Not purchases; a return would otherwise count as a purchase. Counts per rule in `outputs/cleaning_log.csv` |
| 3 | Keep only stock codes of 5 digits + up to 2 letters | Drops POST, M, D, BANK CHARGES, DOT, TEST, gift vouchers; these are fees/adjustments, not products. Stock codes upper-cased so `85123a` = `85123A` |
| 4 | Drop lines with no customer ID (~23% of raw rows) | Cannot be attached to a customer row; they stay out of the matrix |
| 5 | Test period = last 2 months (from 2011-10-09); training = everything before | Time-based split mimics predicting the future (R2) |
| 6 | Held-out = test products a customer never bought in training | Measures discovery of new products rather than reorders (brief DE3) |
| 7 | Cleaning applied before the split, nothing fitted on test | Cleaning rules use no statistics, so no leakage; vocabulary, product filter, popularity and all models use training rows only |
| 8 | Keep products bought by >= 10 training customers | Rarer products have too little signal for stable factors; 3,686 of 4,529 kept |
| 9 | Purchase value log(1 + units), summed per customer-product | Bulk orders (hundreds of units) would dominate raw counts. ML7 later confirmed raw count is clearly worst (13.6% pure SVD, 26.1% hybrid vs 22.5% / 36.6% for log) |
| 10 | Main model: Truncated SVD (decided before fitting) | Sparse input natively, product factors in the same fit, no centring that turns "not bought" into a negative signal |
| 11 | k and alpha chosen on a validation split inside the training period, not on test | R2: the test period must not drive model selection. Rule: smallest k whose validation hit rate is within 3% of the best (k = 100); alpha = best at that k (3.0) |
| 12 | Add a popularity prior: score = sim / max sim + alpha * log-popularity | Pure SVD similarity (22.5%) lost to best sellers (27.0%) in both validation and test. Popularity is a strong signal in reorder-heavy wholesale data. Both versions reported |
| 13 | Run PCA (arpack, implicit centring) with the same k and the same scoring | Fair PCA vs SVD comparison (ML5) |
| 14 | Customer map revised after review: normalize SVD customer factors, project onto 2 PCA components, then fit k-means on those same coordinates; select 3 groups from 3 to 6 by silhouette | Follows ML6's projection-then-clustering sequence. Unit-length factors reduce volume effects. The new groups are Mixed gifts and baking shops, Heart and home-decor shops, and Jumbo-bag buyers. Silhouette is about 0.39 in the 2-D space; this is not directly comparable with the previous 100-dimensional score. The projection simplifies buying patterns and groups remain campaign themes to validate |
| 15 | Customer IDs only in `data/processed/customers.csv` (git-ignored); outputs use group names | R4 |
| 16 | Define best sellers by the number of distinct training customers per product; exclude each customer's previous purchases | Explicit popularity measure, using training data only, for both the baseline and the hybrid prior |
| 17 | Report calculated float64 factor storage against both dense and sparse input storage; label timing as the model stage | 7.33 MB factors are about 22x smaller than 161.3 MB dense storage but larger than the 4.99 MB sparse input. Timing covers fit, all product similarities and 1,913 evaluation customers; loading, preprocessing and saving are excluded |
| 18 | Complete the PCA/SVD table with component-1 interpretations and sampled total process RSS during fit | ML5 requires both measures. Correlations use the sum of log purchase values, not raw units. Total RSS includes data already loaded and allocations retained by earlier computations |
