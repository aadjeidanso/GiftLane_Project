# Emmanuel Adjei Danso - Individual reflection

**What I did:** I built the full pipeline alone: cleaned 1.07M invoice lines, made a leak-free time split, built the sparse customer-by-product matrix, compared Truncated SVD with PCA, built the recommender and evaluation, clustered customers into three named groups, and wrote the memo and slides.

**What I learned:** Compression worked well on size (161 MB to 7 MB) and speed (under a second), but the SVD factors alone lost to a best-seller list (22.5% vs 27.0%) because buyers here mostly reorder popular items. Adding a popularity prior, tuned on validation data rather than the test period, lifted the hit rate to 36.6%. PCA and SVD performed almost the same on this matrix, and PCA's first component was still "overall volume", so centring changed less than I expected.

**What I would change:** Test recency weighting and stock/price filters, check the groups with Marketing, and run a live A/B test, since hit rate on held-out purchases undercounts good suggestions.
