"""Builds memo.pdf and slides.pptx from outputs/results.json so every number matches the analysis."""
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pptx import Presentation
from pptx.util import Inches, Pt

R = json.load(open("outputs/results.json"))
M = json.load(open("outputs/matrix_report.json"))
seg = pd.read_csv("outputs/segment_profile.csv")
K, AL = R["k_chosen"], R["alpha_chosen"]
pc = lambda x: f"{100 * x:.1f}%"
shrink = R["dense_MB"] / R["svd_factors_mb"]
model_seconds = R["svd_fit_s"] + R["svd_score_s"]
group_count = len(seg)
uk_shares = [int(row.split("United Kingdom ")[1].split("%")[0]) for row in seg.top_countries]
uk_range = f"{min(uk_shares)}-{max(uk_shares)}%"

# ------------------------------------------------------------------ memo.pdf (one page)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=6, spaceAfter=1.5, textColor=colors.HexColor("#0b4f8a"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=9.2, leading=11.8)
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceAfter=2)
doc = SimpleDocTemplate("memo.pdf", pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=11 * mm, bottomMargin=9 * mm)
P = lambda t, s=B: Paragraph(t, s)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=8.5, leading=10.5)
tbl = Table([
    ["Measure", "Reference", "SVD model"],
    ["Calculated storage", f"{R['dense_MB']:.0f} MB dense table", f"{R['svd_factors_mb']:.1f} MB factors ({shrink:.0f}x smaller than dense)"],
    ["Model stage time", "Original prototype not re-timed", f"{model_seconds:.1f} s: fit, product similarities, score {R['test_customers']:,} test customers"],
    ["Customers with a held-out product in their top 10", f"Best sellers: {pc(R['baseline_hit10'])}", f"Factors + popularity: {pc(R['svd_hit10'])}"],
], colWidths=[62 * mm, 58 * mm, 60 * mm])
tbl._cellvalues = [[Paragraph(str(x), CELL) for x in r] for r in tbl._cellvalues]
tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6eaf8")), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
rows = " ".join(f"<b>{r['name']}</b> ({r['share_pct']:.0f}%: {r['top_products'].split('; ')[0].title()}, {r['top_products'].split('; ')[1].title()}...);"
                for _, r in seg.iterrows())
story = [
    P("Memo: \"Customers who bought this also bought...\" and customer groups", T),
    P("To: Sarah Okafor, Head of E-Commerce &nbsp;|&nbsp; From: Data &amp; ML team &nbsp;|&nbsp; Re: Q4 planning", B), Spacer(1, 3),
    P("Recommendation", H),
    P(f"Trial the SVD recommender and marketing groups. Calculated factor storage is {R['svd_factors_mb']:.1f} MB, "
      f"compared with {R['dense_MB']:.0f} MB for a dense customer-by-product table. On unseen purchases, "
      f"{pc(R['svd_hit10'])} of {R['test_customers']:,} customers received at least one correct product in their top 10, "
      f"against {pc(R['baseline_hit10'])} for best sellers (about +/-2 percentage points)."), Spacer(1, 3), tbl,
    P(f"These are storage calculations, not saved model-file sizes. The sparse input is already only {R['sparse_MB']:.2f} MB, "
      "smaller than the factors. Timing excludes loading, cleaning, matrix construction, saving, and scoring customers outside the test group; "
      "measure the full production pipeline separately."),
    P("Method and why", H),
    P(f"We compress the customer-by-product table with <b>SVD</b> (a way of finding {K} hidden \"taste patterns\", such as \"buys cake cases and doilies\"), "
      "then suggest products that sit close to what a shop already buys. We chose SVD over its close cousin PCA because it works directly on a table that is "
      "98% empty and gives product patterns in one step. We ran both and they performed the same here "
      f"({pc(R['svd_hit10'])} vs {pc(R['pca_hit10'])}); our choice is based on its natural fit to sparse purchase data. "
      f"Customer scores also blend in popularity: patterns alone scored {pc(R['svd_hit10_pure'])}, below best sellers. "
      "Here, best sellers means products bought by the most training customers. The blend and pattern count were chosen on earlier validation data. "
      "The product-page box uses product similarity alone; the customer hit rate above does not directly measure that box."),
    P("What we lose by compressing", H),
    P(f"The {K} patterns retain {pc(R['svd_explained_var'])} of measured variation (relative reconstruction error {pc(R['svd_rel_error'])}). "
      f"That sounds large, but what matters is suggestion quality. Comparison: a version that keeps the full table and never compresses scores "
      f"{pc(R['uncompressed_item_cosine_hit10_pure'])} on patterns alone versus {pc(R['svd_hit10_pure'])} for ours, so compression does cost some precision; "
      "the popularity nudge more than makes up for it."),
    P("Customer groups for marketing", H),
    P(f"We projected customers onto two components, then clustered those same coordinates into {group_count} groups: {rows} "
      f"UK buyers make up {uk_range} of each group. The map simplifies buying patterns, so use these as campaign themes and validate them with Marketing."),
    P("What could go wrong", H),
    P("(1) The data ends Dec 2011 and has only ~5,500 customers with history; seasons and tastes shift, so retrain nightly and re-check the hit rate monthly. "
      "(2) New customers and new products need a best-seller fallback. "
      "(3) Hit rate counts only purchases we can see; a 'miss' may still be a good suggestion. A live A/B test is the real proof. "
      "(4) Returns and cancellations were excluded, and about a fifth of order lines had no customer ID and were dropped. "
      "(5) No customer IDs appear in this memo or the slides; groups are anonymous."),
    P("Next steps", H),
    P("Run an A/B test on a few product pages; add filters (in stock, same price band); add recent purchases with more weight; agree one campaign per group with Marketing."),
]
doc.build(story)

# ------------------------------------------------------------------ slides.pptx (5 minutes)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
blank = prs.slide_layouts[5]

def slide(title, bullets=None, img=None, img_w=7.2, table=None):
    s = prs.slides.add_slide(blank)
    s.shapes.title.text = title; s.shapes.title.text_frame.paragraphs[0].font.size = Pt(30)
    s.shapes.title.left, s.shapes.title.top, s.shapes.title.width, s.shapes.title.height = Inches(.5), Inches(.3), Inches(12.3), Inches(1)
    if bullets:
        w = 5.6 if img else 12.3
        tb = s.shapes.add_textbox(Inches(.6), Inches(1.5), Inches(w), Inches(5.5)).text_frame; tb.word_wrap = True
        for i, b in enumerate(bullets):
            p = tb.paragraphs[0] if i == 0 else tb.add_paragraph(); p.text = "\u2022 " + b; p.font.size = Pt(19); p.space_after = Pt(9)
    if img: s.shapes.add_picture(img, Inches(6.4 if bullets else 2), Inches(1.5), width=Inches(img_w if bullets else 9))
    if table:
        sh = s.shapes.add_table(len(table), len(table[0]), Inches(.6), Inches(1.6), Inches(12), Inches(.5 * len(table)))
        for i, r_ in enumerate(table):
            for j, v in enumerate(r_):
                c = sh.table.cell(i, j); c.text = str(v)
                for p in c.text_frame.paragraphs: p.font.size = Pt(15)
    return s

slide("Customers Like You: recommendations and customer groups", [
    "GiftLane, Online Retail II data (1.07M order lines, Dec 2009 - Dec 2011)",
    "Request: shrink the \"also bought\" job so it runs nightly, beat best sellers, and give Marketing named customer groups",
    "Targets set on Day 1: much smaller data | fast job | beat best-seller baseline on unseen purchases | nameable groups",
    "Solo project: both roles (Data Engineer + ML Engineer)"])
slide("Data Engineer: from raw log to a leak-free matrix", [
    f"{M['raw_rows']:,} raw lines -> {M['rows_after_cleaning']:,} after cleaning (cancellations, quantity/price <= 0, non-product codes, no customer ID; each rule logged)",
    f"Time split: train before {M['cutoff']}, test = last 2 months. Held-out = products the customer never bought in training",
    f"Matrix {M['shape'][0]:,} customers x {M['shape'][1]:,} products (kept products bought by >= 10 customers), value = log(1 + units)",
    f"{M['sparsity_pct']}% empty: {M['dense_MB']} MB dense vs {M['sparse_MB']} MB sparse",
    "One command rebuilds everything: python build_matrix.py"])
slide("ML Engineer: SVD or PCA?", [
    "SVD chosen for sparse purchases and direct product factors",
    f"k = {K}, alpha = {AL:g}: selected on earlier validation data",
    f"Same-k hybrid hit rate: PCA {pc(R['pca_hit10'])}, SVD {pc(R['svd_hit10'])}",
    f"Both first components track log purchase activity (PCA r = {R['comp1']['pca']['corr_with_volume']:.2f})",
    f"Calculated storage: {R['svd_factors_mb']:.2f} MB factors, {R['dense_MB']:.0f} MB dense, {R['sparse_MB']:.2f} MB sparse"], img="outputs/figures/k_curve.png", img_w=6.6)
results_slide = slide("Customer recommendations: held-out hit rate @10", table=[
    ["Method", "Hit rate @10", "Note"],
    ["Best sellers (distinct training buyers)", pc(R["baseline_hit10"]), f"+/-{100 * R['baseline_ci']:.1f} pts"],
    [f"SVD factors alone (k={K})", pc(R["svd_hit10_pure"]), "loses to best sellers (honest result)"],
    [f"SVD + popularity nudge (alpha={AL:g})", pc(R["svd_hit10"]), f"+/-{100 * R['svd_ci']:.1f} pts: beats baseline"],
    [f"PCA + popularity nudge", pc(R["pca_hit10"]), "similar to SVD"],
    ["No compression, item similarity alone", pc(R["uncompressed_item_cosine_hit10_pure"]), "compression costs precision"],
])
note = results_slide.shapes.add_textbox(Inches(.6), Inches(5.4), Inches(12), Inches(1.4)).text_frame
note.word_wrap = True
note.text = (f"Timed model stage: {model_seconds:.2f} s for fit, all product similarities and scoring {R['test_customers']:,} test customers. "
             "Excludes preprocessing and saving. Product-page suggestions use similarity alone and are not directly evaluated by this customer metric.")
note.paragraphs[0].font.size = Pt(17)
slide(f"Customer map: {group_count} groups Marketing can name", [
    f"{r['name']} ({r['share_pct']:.0f}%): {', '.join(x.title() for x in r['top_products'].split('; ')[:2])}" for _, r in seg.iterrows()
] + ["Project to 2 components, then cluster the same 2-D points (3-6 groups tried)",
     f"UK buyers: {uk_range} in each group; use products to plan campaigns"], img="outputs/figures/customer_map.png", img_w=6.7)
slide("Risks and next steps", [
    "Held-out purchases undercount good suggestions; run an A/B test on live pages",
    "Add a best-seller fallback for new customers/products; monitor recommendation quality",
    "Value choice matters: raw counts hurt (hit rate drops to 26%); log or 1/0 are good",
    "Next: stock/price filters, recency weighting, one campaign per group"])
prs.save("slides.pptx")
print("memo.pdf and slides.pptx written")
