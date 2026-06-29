import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import requests
import re
import json
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Smart Grocery Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 50%, #0a0f1e 100%);
    min-height: 100vh;
}

/* Cards */
.prod-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 3px solid #8b5cf6;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
}
.prod-card:hover {
    transform: translateY(-3px);
    background: rgba(139,92,246,0.08);
    border-color: rgba(139,92,246,0.5);
    box-shadow: 0 8px 32px rgba(139,92,246,0.2);
}
.prod-card-cv {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px; padding: 16px; margin-bottom: 12px;
    text-align: center; backdrop-filter: blur(10px);
    transition: all 0.25s ease;
}
.prod-card-cv:hover {
    transform: translateY(-3px);
    background: rgba(99,102,241,0.08);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}

/* Text */
.prod-name { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1rem; color:#f1f5f9; }
.prod-name-cv { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.9rem; color:#f1f5f9; margin:6px 0 4px 0; }
.prod-meta { font-size:0.8rem; color:#94a3b8; margin-top:5px; }

/* Badges */
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; margin-right:4px; }
.b-cat   { background:rgba(139,92,246,0.15); color:#c4b5fd; border:1px solid rgba(139,92,246,0.3); }
.b-sub   { background:rgba(251,146,60,0.15); color:#fdba74; border:1px solid rgba(251,146,60,0.3); }
.b-price { background:rgba(52,211,153,0.15); color:#6ee7b7; border:1px solid rgba(52,211,153,0.3); }
.b-rank  { background:rgba(99,102,241,0.2); color:#a5b4fc; border:1px solid rgba(99,102,241,0.4); }
.b-algo  { background:linear-gradient(90deg,#7c3aed,#4f46e5); color:white; padding:4px 16px; border-radius:20px; font-size:0.78rem; font-weight:600; box-shadow:0 0 20px rgba(124,58,237,0.4); }
.b-green { background:rgba(52,211,153,0.15); color:#6ee7b7; border:1px solid rgba(52,211,153,0.3); }
.b-conf  { background:rgba(99,102,241,0.1); color:#a5b4fc; font-size:0.65rem; border:1px solid rgba(99,102,241,0.2); }

.tag { background:rgba(139,92,246,0.1); color:#c4b5fd; padding:2px 8px; border-radius:8px; font-size:0.7rem; margin-right:3px; display:inline-block; border:1px solid rgba(139,92,246,0.2); }

/* Metric boxes */
.mbox {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px; padding: 20px 14px; text-align:center;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
}
.mbox:hover { border-color: rgba(139,92,246,0.5); background: rgba(139,92,246,0.06); }
.mnum { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; background:linear-gradient(135deg,#a78bfa,#6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.mlbl { font-size:0.75rem; color:#64748b; margin-top:4px; letter-spacing:0.05em; text-transform:uppercase; }

/* Section headers */
.sec-hdr { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; color:#e2e8f0;
           border-bottom:1px solid rgba(139,92,246,0.3); padding-bottom:8px; margin-bottom:14px; }
.sec-hdr-green { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; color:#e2e8f0;
                 border-bottom:1px solid rgba(52,211,153,0.4); padding-bottom:8px; margin-bottom:14px; }

/* Eval cards */
.eval-card {
    background: rgba(255,255,255,0.03); border:1px solid rgba(139,92,246,0.2);
    border-radius:14px; padding:20px; margin-bottom:12px;
    border-top:2px solid #7c3aed; backdrop-filter:blur(10px);
}
.eval-val  { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; color:#e2e8f0; }
.eval-lbl  { font-size:0.8rem; color:#94a3b8; margin-top:4px; }
.eval-hint { font-size:0.72rem; color:#475569; margin-top:4px; }

/* Progress bars */
.prog-wrap { background:rgba(255,255,255,0.06); border-radius:10px; height:8px; margin:4px 0; overflow:hidden; }
.prog-fill  { height:8px; border-radius:10px; background:linear-gradient(90deg,#7c3aed,#4f46e5); }
.conf-bar-wrap { background:rgba(52,211,153,0.1); border-radius:6px; height:5px; margin:3px 0 6px 0; }
.conf-bar { background:linear-gradient(90deg,#34d399,#059669); height:5px; border-radius:6px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0a1a 0%,#0d0d2b 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.15);
}
[data-testid="stSidebar"] label { color:#94a3b8 !important; font-size:0.85rem; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label { color:#94a3b8 !important; }
[data-testid="stSidebar"] p { color:#64748b !important; }

/* Misc */
.search-hit {
    background: rgba(255,255,255,0.03); border-radius:12px; padding:12px 16px; margin-bottom:8px;
    cursor:pointer; border-left:3px solid #7c3aed;
    border: 1px solid rgba(139,92,246,0.2);
}
.cs-card {
    background: linear-gradient(135deg,rgba(124,58,237,0.3),rgba(79,70,229,0.3));
    border:1px solid rgba(139,92,246,0.4);
    border-radius:16px; padding:20px; color:white; margin-bottom:16px;
    backdrop-filter:blur(10px);
}
.cv-banner {
    background: linear-gradient(135deg,rgba(10,10,26,0.9),rgba(13,13,43,0.9));
    border:1px solid rgba(52,211,153,0.4); border-radius:12px; padding:1rem 1.5rem;
    margin:1rem 0; color:white; backdrop-filter:blur(10px);
}
.img-preview {
    border:2px dashed rgba(52,211,153,0.4); border-radius:12px; padding:0.5rem;
    background:rgba(52,211,153,0.03); text-align:center; margin-bottom:0.75rem;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius:12px; padding:12px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #7c3aed !important; }
.stDataFrame { background: rgba(255,255,255,0.02) !important; }
p, li, span { color: #94a3b8; }
h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
import os

def find_file(filename):
    """Search for CSV in common Streamlit Cloud + local paths."""
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"❌ '{filename}' not found! Make sure it's committed to your GitHub repo "
        f"in the same folder as app.py (or in a 'data/' subfolder).\n"
        f"Searched: {candidates}"
    )

@st.cache_data
def load_data():
    products = pd.read_csv(find_file("products_500plus.csv"))
    ratings  = pd.read_csv(find_file("user_ratings.csv"))
    return products, ratings

@st.cache_data
def build_all_models(ratings_hash):
    ratings  = pd.read_csv(find_file("user_ratings.csv"))
    products = pd.read_csv(find_file("products_500plus.csv"))
    pivot    = ratings.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    mat      = pivot.values.astype(float)

    user_sim_mat = cosine_similarity(csr_matrix(mat))
    item_sim_mat = cosine_similarity(csr_matrix(mat.T))
    user_sim_df  = pd.DataFrame(user_sim_mat, index=pivot.index, columns=pivot.index)
    item_sim_df  = pd.DataFrame(item_sim_mat, index=pivot.columns, columns=pivot.columns)

    k = min(20, min(mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(mat), k=k)
    predicted    = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred_df      = pd.DataFrame(predicted, index=pivot.index, columns=pivot.columns)

    pop = ratings.groupby('product_id').agg(
        count=('rating','count'), avg_rating=('rating','mean')
    ).reset_index()
    pop['pop_score'] = pop['count'] * pop['avg_rating']
    pop = pop.sort_values('pop_score', ascending=False)

    return pivot, user_sim_df, item_sim_df, pred_df, pop

@st.cache_data
def compute_eval_metrics():
    ratings  = pd.read_csv(find_file("user_ratings.csv"))
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=42)
    train_pivot = train_r.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    train_mat   = train_pivot.values.astype(float)

    k = min(20, min(train_mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(train_mat), k=k)
    pred_mat      = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred_train_df = pd.DataFrame(pred_mat, index=train_pivot.index, columns=train_pivot.columns)

    test_pivot = test_r.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    cu = [u for u in test_pivot.index if u in pred_train_df.index]
    cp = [p for p in test_pivot.columns if p in pred_train_df.columns]
    actual_flat = test_pivot.loc[cu, cp].values.flatten()
    pred_flat   = pred_train_df.loc[cu, cp].values.flatten()
    mask        = actual_flat > 0
    rmse_svd    = float(np.sqrt(mean_squared_error(actual_flat[mask], pred_flat[mask])))

    user_sim    = cosine_similarity(csr_matrix(train_mat))
    user_sim_df2 = pd.DataFrame(user_sim, index=train_pivot.index, columns=train_pivot.index)
    ub_preds, ub_actuals = [], []
    for _, row in test_r.iterrows():
        uid, pid, actual = row['user_id'], row['product_id'], row['rating']
        if uid not in user_sim_df2.index or pid not in train_pivot.columns: continue
        sim_scores = user_sim_df2[uid].drop(uid).sort_values(ascending=False).head(15)
        numer = sum(sim_scores.get(n, 0) * train_pivot.loc[n, pid] for n in sim_scores.index if train_pivot.loc[n, pid] > 0)
        denom = sum(abs(sim_scores.get(n, 0)) for n in sim_scores.index if train_pivot.loc[n, pid] > 0)
        if denom > 0:
            ub_preds.append(numer / denom)
            ub_actuals.append(actual)
    rmse_ub = float(np.sqrt(mean_squared_error(ub_actuals, ub_preds))) if ub_preds else 0.0

    K = 10
    precisions, recalls = [], []
    threshold   = 3.5
    test_grouped = test_r[test_r['rating'] >= threshold].groupby('user_id')['product_id'].apply(set).to_dict()
    for uid in list(test_grouped.keys())[:50]:
        if uid not in pred_train_df.index: continue
        rated_train = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
        preds_uid   = pred_train_df.loc[uid].drop(list(rated_train), errors='ignore')
        top_k       = set(preds_uid.sort_values(ascending=False).head(K).index)
        relevant    = test_grouped.get(uid, set())
        hits        = top_k & relevant
        precisions.append(len(hits) / K)
        recalls.append(len(hits) / len(relevant) if relevant else 0)

    p_at_k = float(np.mean(precisions)) if precisions else 0.0
    r_at_k = float(np.mean(recalls))    if recalls    else 0.0
    f1     = float(2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0

    all_recommended = set()
    for uid in list(pred_train_df.index)[:50]:
        rated = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
        top   = pred_train_df.loc[uid].drop(list(rated), errors='ignore').sort_values(ascending=False).head(K).index
        all_recommended.update(top)
    coverage = float(len(all_recommended) / len(train_pivot.columns))

    return {'rmse_svd': rmse_svd, 'rmse_ub': rmse_ub,
            'precision_at_k': p_at_k, 'recall_at_k': r_at_k,
            'f1': f1, 'coverage': coverage, 'K': K}


products_df, ratings = load_data()
ratings_hash = len(ratings)
pivot, user_sim_df, item_sim_df, pred_df, pop_df = build_all_models(ratings_hash)
product_map  = products_df.set_index('product_id').to_dict('index')
all_users    = sorted(pivot.index.tolist())
all_pids     = sorted(products_df['product_id'].tolist())
categories   = sorted(products_df['category'].unique().tolist())


# ─────────────────────────────────────────────
# CF RECOMMENDER FUNCTIONS
# ─────────────────────────────────────────────
def user_based_recommend(user_id, top_n=10, n_neighbors=15, cat_filter=None):
    if user_id not in user_sim_df.index: return []
    sim_scores = user_sim_df[user_id].drop(user_id).sort_values(ascending=False).head(n_neighbors)
    rated      = set(pivot.loc[user_id][pivot.loc[user_id] > 0].index)
    scores = {}
    for nb in sim_scores.index:
        w = sim_scores[nb]
        for pid, r in pivot.loc[nb].items():
            if r > 0 and pid not in rated:
                if cat_filter and product_map.get(pid, {}).get('category') not in cat_filter: continue
                scores[pid] = scores.get(pid, 0) + w * r
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def item_based_recommend(user_id, top_n=10, cat_filter=None):
    if user_id not in pivot.index: return []
    user_ratings = pivot.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    if rated.empty: return []
    scores = {}
    for pid, r in rated.items():
        if pid not in item_sim_df.index: continue
        for other, sim in item_sim_df[pid].drop(pid).items():
            if other not in rated.index:
                if cat_filter and product_map.get(other, {}).get('category') not in cat_filter: continue
                scores[other] = scores.get(other, 0) + sim * r
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def svd_recommend(user_id, top_n=10, cat_filter=None):
    if user_id not in pred_df.index: return []
    rated = set(pivot.loc[user_id][pivot.loc[user_id] > 0].index)
    preds = pred_df.loc[user_id].drop(list(rated), errors='ignore')
    if cat_filter:
        preds = preds[[p for p in preds.index if product_map.get(p, {}).get('category') in cat_filter]]
    return preds.sort_values(ascending=False).head(top_n).index.tolist()

def hybrid_recommend(user_id, top_n=10, alpha=0.4, beta=0.35, cat_filter=None):
    ub  = user_based_recommend(user_id, top_n=top_n*3, cat_filter=cat_filter)
    ib  = item_based_recommend(user_id, top_n=top_n*3, cat_filter=cat_filter)
    svd = svd_recommend(user_id,        top_n=top_n*3, cat_filter=cat_filter)
    scores = {}
    for rank, pid in enumerate(ub):
        scores[pid] = scores.get(pid, 0) + alpha * (1 / (rank + 1))
    for rank, pid in enumerate(ib):
        scores[pid] = scores.get(pid, 0) + beta * (1 / (rank + 1))
    for rank, pid in enumerate(svd):
        g = 1 - alpha - beta
        scores[pid] = scores.get(pid, 0) + g * (1 / (rank + 1))
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def popularity_recommend(top_n=10, cat_filter=None):
    df = pop_df.copy()
    if cat_filter:
        df = df[df['product_id'].map(lambda p: product_map.get(p, {}).get('category')).isin(cat_filter)]
    return df.head(top_n)['product_id'].tolist()

def cold_start_recommend(selected_cats, top_n=10):
    df = pop_df.copy()
    df['cat'] = df['product_id'].map(lambda p: product_map.get(p, {}).get('category'))
    return df[df['cat'].isin(selected_cats)].head(top_n)['product_id'].tolist()

def similar_items_cf(pid, top_n=8, cat_filter=None):
    if pid not in item_sim_df.index: return []
    sims = item_sim_df[pid].drop(pid).sort_values(ascending=False)
    if cat_filter:
        sims = sims[[p for p in sims.index if product_map.get(p, {}).get('category') in cat_filter]]
    return sims.head(top_n).index.tolist()


# ─────────────────────────────────────────────
# IMAGE SCANNER — KEYWORD & MAPPING DATA
# ─────────────────────────────────────────────
RELATED_CATEGORIES = {
    "Drinks":        ["Drinks", "Beverages"],
    "Beverages":     ["Beverages", "Drinks", "Health"],
    "Snacks":        ["Snacks", "Bakery", "Frozen"],
    "Bakery":        ["Bakery", "Snacks", "Dairy", "Condiments"],
    "Dairy":         ["Dairy", "Bakery", "Condiments", "Beverages"],
    "Grains":        ["Grains", "Spices", "Condiments", "Noodles"],
    "Spices":        ["Spices", "Grains", "Condiments", "Noodles"],
    "Noodles":       ["Noodles", "Grains", "Spices", "Condiments", "Snacks"],
    "Condiments":    ["Condiments", "Spices", "Grains", "Noodles", "Bakery"],
    "Personal Care": ["Personal Care", "Home Care", "Health"],
    "Health":        ["Health", "Personal Care", "Beverages", "Grains"],
    "Home Care":     ["Home Care", "Personal Care"],
    "Frozen":        ["Frozen", "Snacks", "Grains", "Noodles"],
}

JUNK_DESCRIPTORS = {
    "block","slab","rectangular","foil","wrapped","solid","object","thing",
    "item","material","texture","surface","background","pattern","shape",
    "band","bandage","adhesive","band aid","medical","first aid","plaster",
    "strip","ribbon","tape","yellow","red","green","white","orange","brown",
    "golden","black","blue","purple","pink","light","dark","bright","shiny",
    "glossy","matte","pale","beige","ivory","small","large","big","round",
    "square","flat","thick","thin","packaging","wrapper","label","sticker",
    "logo","fat","spread","saturated","mix","mixed","blend","product",
    "ingredient","ingredients","donut","doughnut","pretzel","ring","coil",
    "spiral","loop","oval","circle","fried","grilled","boiled","baked",
    "roasted","steamed","curved","elongated","crescent","arc","peel","skin","bunch",
}

CUISINE_NOISE = {
    "greek","mediterranean","italian","chinese cuisine","mexican","thai cuisine",
    "japanese cuisine","korean","american","continental","fusion","ethnic",
    "dish","meal","cuisine","platter","recipe","delicacy","homemade","restaurant",
}

DAIRY_SPECIFIC = {
    "butter": {
        "tags":   {"butter","amul butter","salted butter","unsalted butter","table butter","cooking butter","makhan","margarine"},
        "pids":   ["P021","P181","P182","P183"],
        "blocks": {"cream","yogurt","curd","dahi","milk","lassi","paneer","cheese","ghee","condensed milk"},
    },
    "ghee": {
        "tags":   {"ghee","clarified butter","desi ghee","pure ghee"},
        "pids":   ["P025","P192","P193"],
        "blocks": {"butter","cream","yogurt","curd","milk","paneer","cheese","condensed milk"},
    },
    "curd": {
        "tags":   {"curd","dahi","yogurt","yoghurt","greek yogurt","set curd","thick curd"},
        "pids":   ["P023","P187","P188","P189","P402","P403","P491"],
        "blocks": {"butter","cream","milk","paneer","cheese","ghee","condensed milk"},
    },
    "milk": {
        "tags":   {"milk","toned milk","full cream milk","skimmed milk","double toned milk"},
        "pids":   ["P194","P195","P196","P401","P404","P405"],
        "blocks": {"butter","cream","curd","yogurt","paneer","cheese","ghee","condensed milk"},
    },
    "cream": {
        "tags":   {"fresh cream","whipping cream","whipped cream","cooking cream"},
        "pids":   ["P197","P198"],
        "blocks": {"butter","curd","yogurt","paneer","cheese","ghee","milk"},
    },
    "cheese": {
        "tags":   {"cheese","processed cheese","cream cheese","cheddar","cheese slice","cheese slices"},
        "pids":   ["P022","P184","P185","P186","P406"],
        "blocks": {"butter","curd","yogurt","cream","ghee","milk","paneer"},
    },
    "paneer": {
        "tags":   {"paneer","cottage cheese"},
        "pids":   ["P024","P190","P191","P499"],
        "blocks": {"butter","curd","yogurt","cream","ghee","milk","cheese"},
    },
}

GROCERY_KEYWORDS = {
    "biscuit":   ["P001","P002","P003","P004","P005","P006","P007","P008","P009","P010","P132","P133","P134","P135","P160","P161","P162","P163","P369","P370","P371","P372"],
    "parle":     ["P001","P010","P157","P162","P475"],
    "britannia": ["P003","P132","P133","P134","P151","P153","P156","P369"],
    "marie":     ["P003"],
    "bourbon":   ["P004"],
    "cookie":    ["P006","P158","P159","P164"],
    "digestive": ["P007","P160"],
    "bread":     ["P153","P154","P155","P471","P472"],
    "cracker":   ["P008","P009"],
    "rusk":      ["P156","P157","P475"],
    "cake":      ["P151","P152","P473","P474"],
    "muffin":    ["P473"],
    "multigrain bread": ["P155"],
    "oreo":      ["P372"],
    "dark fantasy": ["P371","P133"],
    "bakery":    ["P001","P002","P003","P004","P006","P007","P151","P152","P153"],
    "chips":     ["P011","P012","P013","P014","P143","P144","P168","P169","P170","P177","P178","P179"],
    "lays":      ["P011","P143"],
    "kurkure":   ["P012"],
    "bingo":     ["P013","P169"],
    "pringles":  ["P014"],
    "namkeen":   ["P015","P016","P017","P141","P142","P171","P172","P173","P174","P175"],
    "haldiram":  ["P015","P016","P017","P112","P141","P171","P172","P173","P343","P444","P461"],
    "bhujia":    ["P015","P173","P174"],
    "mixture":   ["P016"],
    "sev":       ["P141","P171"],
    "boondi":    ["P142"],
    "popcorn":   ["P165","P166"],
    "nacho":     ["P144","P168","P176"],
    "doritos":   ["P144"],
    "peanut":    ["P167"],
    "makhana":   ["P180"],
    "papad":     ["P393","P394"],
    "wafers":    ["P177","P178","P179"],
    "chocolate bar": ["P373","P374","P375","P376","P463","P495","P496"],
    "candy":     ["P464","P465","P466","P467","P468"],
    "snack":     ["P011","P012","P013","P015","P016","P165","P166"],
    "butter":    ["P021","P181","P182","P183"],
    "amul butter": ["P021","P181","P182"],
    "amul":      ["P021","P023","P025","P026","P027","P181","P182","P183","P184","P185","P186","P187","P188","P189","P190","P191","P192","P193","P194","P195","P196","P197","P198","P199","P200","P201"],
    "ghee":      ["P025","P192","P193"],
    "curd":      ["P023","P187","P188","P189","P402","P403"],
    "yogurt":    ["P023","P187","P188","P189","P491"],
    "dahi":      ["P023","P187","P188","P402","P403"],
    "paneer":    ["P024","P190","P191","P499"],
    "cheese":    ["P022","P184","P185","P186","P406"],
    "milk":      ["P194","P195","P196","P401","P404","P405"],
    "cream":     ["P197","P198"],
    "lassi":     ["P136","P199","P200","P500"],
    "shrikhand": ["P026","P201"],
    "condensed milk": ["P027","P198"],
    "ice cream": ["P115","P116","P347","P348","P349","P350","P351","P440","P441","P445"],
    "kulfi":     ["P445"],
    "gulab jamun": ["P114","P202","P351","P444"],
    "dairy":     ["P021","P022","P023","P024","P025","P194","P195","P196"],
    "rice":      ["P031","P032","P203","P204","P407","P408","P492"],
    "basmati":   ["P031","P032","P203","P204","P407","P408","P492"],
    "dal":       ["P033","P034","P205","P206","P207","P208","P377"],
    "toor dal":  ["P033","P205"],
    "chana dal": ["P034","P206"],
    "moong dal": ["P208","P017"],
    "masoor dal": ["P207"],
    "urad dal":  ["P377"],
    "atta":      ["P035","P145","P209","P217"],
    "wheat flour": ["P035","P145","P209"],
    "maida":     ["P036","P210"],
    "besan":     ["P211","P378"],
    "oats":      ["P037","P038","P094","P146","P212","P213","P214","P411","P412"],
    "poha":      ["P039","P215","P409","P413"],
    "suji":      ["P040","P216","P414"],
    "soya":      ["P218"],
    "salt":      ["P219","P493","P494","P498"],
    "grains":    ["P031","P032","P033","P034","P035"],
    "masala":    ["P041","P042","P045","P047","P048","P147","P148","P220","P221","P222","P223","P224","P225","P234","P235","P381","P415","P416","P497"],
    "mdh":       ["P041","P045","P147","P220","P221","P222","P234","P381","P416","P497"],
    "everest":   ["P042","P044","P046","P048","P148","P223","P224","P225","P228","P235","P415"],
    "garam masala": ["P041","P222"],
    "chilli powder": ["P044","P227"],
    "turmeric":  ["P043","P226"],
    "haldi":     ["P043","P226"],
    "coriander powder": ["P046","P228"],
    "rajma masala": ["P045","P223"],
    "chana masala": ["P047"],
    "pav bhaji": ["P048","P224"],
    "biryani masala": ["P147","P225"],
    "cooking oil": ["P049","P050","P230","P231","P232","P233","P379","P380","P399","P400","P489"],
    "sunflower oil": ["P050","P230"],
    "olive oil": ["P233","P380","P399","P489"],
    "mustard oil": ["P232","P379"],
    "spices":    ["P041","P042","P043","P044","P226","P227","P228"],
    "noodles":   ["P051","P052","P053","P054","P131","P236","P237","P238","P239","P240","P241","P242","P384"],
    "maggi":     ["P051","P131","P236","P237"],
    "yippee":    ["P052","P238","P239"],
    "pasta":     ["P055","P056","P243","P244","P245","P385"],
    "hakka noodles": ["P240"],
    "wai wai":   ["P054","P241"],
    "knorr":     ["P053","P242","P383"],
    "cup noodles": ["P051","P052","P131"],
    "instant noodles": ["P051","P052","P053","P054","P131","P236","P237","P238","P384"],
    "soup":      ["P053","P383","P479"],
    "juice":     ["P061","P062","P063","P066","P067","P137","P138","P246","P247","P248","P249","P250","P386","P387","P396","P397","P423","P424","P469","P470"],
    "mango juice": ["P062","P063","P066","P246","P249","P396"],
    "orange juice": ["P061","P248"],
    "frooti":    ["P063","P249"],
    "maaza":     ["P066"],
    "tropicana": ["P061","P138","P248"],
    "real juice": ["P062","P137","P246","P247","P386","P387"],
    "paper boat": ["P067","P396","P397","P470"],
    "pomegranate juice": ["P137","P387"],
    "guava juice": ["P138","P386"],
    "apple soda": ["P139","P250"],
    "appy":      ["P139","P250"],
    "energy drink": ["P064","P070","P251","P252","P253"],
    "red bull":  ["P070","P251"],
    "sting":     ["P064","P252"],
    "monster":   ["P253"],
    "soda":      ["P065","P139","P140","P254","P255","P256","P257","P258","P388","P389","P398"],
    "limca":     ["P065","P254"],
    "thums up":  ["P255"],
    "sprite":    ["P256"],
    "coca cola": ["P257","P398"],
    "pepsi":     ["P258"],
    "mountain dew": ["P388"],
    "fanta":     ["P389"],
    "water":     ["P069","P259","P260","P390","P490"],
    "bisleri":   ["P069","P259"],
    "mango":     ["P062","P063","P066","P246","P249","P396"],
    "drinks":    ["P061","P062","P063","P064","P065","P066"],
    "jam":       ["P071","P261","P262","P263","P391"],
    "kissan":    ["P071","P261","P262"],
    "ketchup":   ["P073","P266","P267","P268"],
    "heinz":     ["P073","P267"],
    "sauce":     ["P072","P073","P075","P264","P265","P277","P392"],
    "chutney":   ["P074","P269","P270"],
    "schezwan":  ["P074","P264"],
    "pickle":    ["P072","P269","P270"],
    "mayonnaise": ["P076","P271","P272"],
    "mayo":      ["P076","P271","P272"],
    "honey":     ["P077","P096","P273","P274"],
    "dabur honey": ["P077","P273"],
    "nutella":   ["P078","P275"],
    "peanut butter": ["P079","P276"],
    "condiments": ["P071","P073","P076","P077","P078","P079"],
    "tea":       ["P121","P122","P125","P126","P130","P352","P353","P354","P355","P356","P357","P419","P420"],
    "tata tea":  ["P121","P352"],
    "red label": ["P122","P353"],
    "green tea": ["P125","P356","P419"],
    "masala chai": ["P126","P354","P420"],
    "coffee":    ["P123","P124","P127","P358","P359","P360","P361","P395","P421","P422"],
    "nescafe":   ["P123","P358","P361","P421"],
    "bru":       ["P124","P359","P422"],
    "bournvita": ["P128"],
    "milo":      ["P129"],
    "horlicks":  ["P091","P306","P367"],
    "complan":   ["P092","P307"],
    "health drink": ["P091","P092","P093","P128","P129","P306","P307","P308","P366","P367","P368"],
    "toothpaste": ["P081","P278","P279","P280","P459"],
    "colgate":   ["P081","P278","P282"],
    "toothbrush": ["P082","P281","P282"],
    "soap":      ["P083","P084","P149","P283","P284","P285","P286","P287"],
    "dove":      ["P083","P285","P289"],
    "dettol":    ["P084","P283","P436","P485","P486"],
    "shampoo":   ["P085","P086","P288","P289","P290","P452","P453"],
    "body lotion": ["P087","P294"],
    "nivea":     ["P087","P294"],
    "coconut oil": ["P088","P291"],
    "parachute": ["P088","P291"],
    "razor":     ["P089","P299","P454"],
    "gillette":  ["P089","P299","P454"],
    "face wash": ["P085","P150","P295","P296","P447","P448","P449"],
    "garnier":   ["P150","P449"],
    "deo":       ["P300","P301","P455","P482","P483"],
    "hair oil":  ["P088","P291","P292","P480","P488"],
    "personal care": ["P081","P082","P083","P084","P085","P086"],
    "detergent": ["P101","P102","P321","P322","P323","P324","P325","P432","P433"],
    "surf excel": ["P101","P321","P432"],
    "ariel":     ["P102","P322","P433"],
    "dishwash":  ["P103","P326","P327","P328","P434"],
    "vim":       ["P103","P326","P434"],
    "toilet cleaner": ["P104","P329","P330","P431"],
    "harpic":    ["P104","P329","P431"],
    "floor cleaner": ["P106","P331","P339"],
    "lizol":     ["P106","P331","P339"],
    "glass cleaner": ["P105","P332"],
    "mosquito":  ["P109","P110","P334","P335","P336","P435"],
    "freshener": ["P107","P337","P338","P438","P439"],
    "scrub":     ["P108","P333","P437"],
    "home care": ["P101","P102","P103","P104","P106","P107"],
    "muesli":    ["P094","P313"],
    "chyawanprash": ["P095","P314"],
    "patanjali": ["P059","P095","P193","P229","P232","P274","P315","P408","P427","P428"],
    "dabur":     ["P077","P096","P273","P292","P314","P363"],
    "vitamin":   ["P097","P319","P320","P362"],
    "corn flakes": ["P311"],
    "kelloggs":  ["P311","P312","P410"],
    "protein":   ["P093","P308","P309","P310"],
    "oatmeal":   ["P037","P038","P212","P213","P214"],
    "supplement": ["P091","P092","P093","P097","P308","P309","P310"],
    "frozen fries": ["P111","P340","P425"],
    "mccain":    ["P111","P340","P425","P446"],
    "samosa":    ["P343"],
    "momo":      ["P341"],
    "nuggets":   ["P443"],
    "frozen":    ["P111","P112","P113","P114","P115","P116","P340","P341","P342","P343","P344"],
    "ready meal": ["P112","P113","P344","P345","P346"],
    "biryani":   ["P147","P225","P346"],
    "paratha":   ["P344"],
}

LOW_PRIORITY_TAGS = {
    "food","bottle","grain","cereal","beverage","juice","sweet","grocery","groceries",
    "food item","indian","staple","packaged","packaged food","processed food",
    "dairy product","dairy products","cooking ingredient","indian food",
    "breakfast","ready to eat","condiment","fruit juice","staples","drink",
}

VISUAL_LABEL_TO_TAG = {
    "butter":"butter","margarine":"butter","ghee":"ghee",
    "curd":"curd","yogurt":"yogurt","yoghurt":"yogurt","greek yogurt":"yogurt",
    "cheese":"cheese","paneer":"paneer","cottage cheese":"paneer",
    "milk":"milk","fresh cream":"cream","whipping cream":"cream",
    "lassi":"lassi","buttermilk":"lassi",
    "ice cream":"ice cream","gelato":"ice cream","kulfi":"kulfi",
    "condensed milk":"condensed milk",
    "biscuit":"biscuit","cookie":"biscuit","cracker":"biscuit","wafer":"biscuit",
    "bread":"bread","loaf":"bread","toast":"bread","bun":"bread",
    "roti":"bread","chapati":"bread","paratha":"bread","naan":"bread",
    "cake":"cake","muffin":"muffin","rusk":"rusk",
    "potato chip":"chips","potato chips":"chips","chip":"chips","crisp":"chips",
    "crisps":"chips","tortilla chip":"nacho","corn chip":"chips",
    "popcorn":"popcorn","bhujia":"bhujia","namkeen":"namkeen","boondi":"boondi",
    "french fries":"frozen fries","nachos":"nacho","wafer":"wafers",
    "chocolate bar":"chocolate bar","candy":"candy","toffee":"candy","gum":"candy",
    "noodle":"noodles","noodles":"noodles","ramen":"noodles","chow mein":"noodles",
    "lo mein":"noodles","spaghetti":"noodles","pasta":"pasta","penne":"pasta",
    "macaroni":"pasta","fusilli":"pasta","vermicelli":"noodles",
    "maggi":"maggi","instant noodles":"maggi","cup noodles":"maggi",
    "rice":"rice","basmati rice":"basmati","dal":"dal","lentil":"dal","lentils":"dal",
    "oatmeal":"oats","porridge":"oats","oats":"oats","granola":"muesli","muesli":"muesli",
    "poha":"poha","upma":"suji","flour":"atta","wheat flour":"atta","atta":"atta",
    "maida":"maida","besan":"besan","salt":"salt",
    "turmeric":"turmeric","chilli":"chilli powder","chili":"chilli powder",
    "masala":"masala","curry powder":"masala","garam masala":"garam masala",
    "cooking oil":"cooking oil","sunflower oil":"sunflower oil","olive oil":"olive oil",
    "mustard oil":"mustard oil",
    "ketchup":"ketchup","mayonnaise":"mayonnaise","honey":"honey","jam":"jam",
    "jelly":"jam","peanut butter":"peanut butter","nutella":"nutella",
    "chutney":"chutney","pickle":"pickle","sauce":"sauce",
    "mango juice":"mango juice","orange juice":"orange juice","energy drink":"energy drink",
    "lemonade":"soda","soda":"soda","water":"water","cold drink":"soda",
    "fruit juice":"juice",
    "tea":"tea","chai":"tea","green tea":"green tea","coffee":"coffee","espresso":"coffee",
    "bournvita":"bournvita","horlicks":"horlicks","milo":"milo",
    "soap":"soap","shampoo":"shampoo","toothpaste":"toothpaste","toothbrush":"toothbrush",
    "face wash":"face wash","razor":"razor","body lotion":"body lotion",
    "deo":"deo","deodorant":"deo","hair oil":"hair oil","coconut oil":"coconut oil",
    "detergent":"detergent","dishwash":"dishwash","toilet cleaner":"toilet cleaner",
    "floor cleaner":"floor cleaner","glass cleaner":"glass cleaner",
    "mosquito repellent":"mosquito","air freshener":"freshener","scrub":"scrub",
    "supplement":"supplement","protein":"protein","vitamin":"vitamin",
    "corn flakes":"corn flakes","muesli":"muesli","chyawanprash":"chyawanprash",
    "frozen fries":"frozen fries","samosa":"samosa","momo":"momo","nuggets":"nuggets",
    "ice cream":"ice cream","kulfi":"kulfi","biryani":"biryani","paratha":"paratha",
    "tube":"toothpaste","powder":"masala","syrup":"honey",
}

GEMINI_PROMPT = """You are an expert Indian grocery product identifier.

Look at this image and identify the grocery/food/household product shown.
Return ONLY a valid JSON array with 4-8 tags. No explanation, no markdown, just raw JSON.

FORMAT: [{"tag": "product_name", "confidence": 90}, ...]

RULES:
1. Brand name first if visible (Amul, Britannia, Parle, Nestlé, Maggi, Haldiram, MDH, Everest, Tata, Lays, Kurkure, Tropicana, Real, Frooti, Dabur, Patanjali, Colgate, Dove, Surf Excel, etc.)
2. Then specific product type
3. Then category

DAIRY (strict rules — each product is different):
- Rectangular solid block in paper/foil wrapper → "butter"
- White/cream liquid in packet or bottle → "milk"
- Semi-solid white substance in cup or tub → "curd" or "yogurt"
- Thick golden/amber liquid in jar or tin → "ghee"
- Soft white block submerged in water → "paneer"
- Pale yellow firm slices or block → "cheese"
- White frozen dessert in cone/cup → "ice cream"

NOODLES (strict):
- Flat square packet with dry cake → "instant noodles" or "maggi"
- NEVER call noodles a donut, ring, pretzel, or coil
- Orange Nestlé packet → "maggi"

SNACKS:
- Metallic pouch with potato slices → "chips" or "lays"
- Puffed corn curls → "kurkure"
- Round tin with stacked chips → "pringles"

ALLOWED TAGS: butter, ghee, curd, yogurt, milk, cream, paneer, cheese, lassi, ice cream,
kulfi, biscuit, bread, rusk, cake, cookie, chips, lays, kurkure, pringles, namkeen,
bhujia, sev, boondi, nacho, popcorn, chocolate bar, candy, noodles, maggi, yippee,
pasta, rice, basmati, dal, toor dal, chana dal, atta, maida, besan, oats, poha, suji,
salt, masala, turmeric, chilli powder, garam masala, cooking oil, sunflower oil,
olive oil, mustard oil, ketchup, jam, honey, mayonnaise, peanut butter, nutella,
chutney, pickle, sauce, juice, mango juice, orange juice, energy drink, soda, water,
tea, green tea, coffee, bournvita, horlicks, soap, shampoo, toothpaste, toothbrush,
face wash, razor, body lotion, deo, hair oil, coconut oil, detergent, dishwash,
toilet cleaner, floor cleaner, mosquito, freshener, supplement, vitamin, corn flakes,
muesli, frozen fries, samosa, momo, nuggets, biryani, paratha

FORBIDDEN TAGS: block, slab, foil, yellow, red, white, blue, strip, ribbon, rectangular,
solid, packaging, curved, elongated, crescent, peel, donut, ring, coil, spiral, fried,
grilled, baked, paella, risotto, oval

confidence: integer 0-100. Return ONLY the JSON array."""


def normalize_tag(tag: str) -> str:
    return " ".join(tag.lower().strip().split())


def detect_dairy_type(tag_texts):
    priority = ["butter","ghee","paneer","cheese","curd","cream","milk"]
    found = {}
    for tag in tag_texts:
        for dtype, info in DAIRY_SPECIFIC.items():
            for kt in info["tags"]:
                if kt == tag or kt in tag or tag in kt:
                    found[dtype] = info
                    break
    for dtype in priority:
        if dtype in found:
            return dtype, found[dtype]["pids"], found[dtype]["blocks"]
    return None, [], set()


def find_products_from_tags(tag_dicts, product_map):
    matched_high = {}
    matched_low  = set()
    raw_tags = []
    for item in tag_dicts:
        t    = item["tag"] if isinstance(item, dict) else str(item)
        conf = float(item.get("confidence", 50)) if isinstance(item, dict) else 50.0
        raw_tags.append((normalize_tag(t), conf))

    tag_texts = [t for t, _ in raw_tags]
    dairy_type, forced_pids, blocked_tags = detect_dairy_type(tag_texts)
    for pid in forced_pids:
        if pid in product_map:
            matched_high[pid] = matched_high.get(pid, 0) + 200

    for (tag, conf) in raw_tags:
        if tag in JUNK_DESCRIPTORS or tag in CUISINE_NOISE: continue
        if dairy_type and tag in blocked_tags: continue
        tag_words = [w for w in tag.split() if len(w) >= 3]
        for keyword, pids in GROCERY_KEYWORDS.items():
            kw = normalize_tag(keyword)
            kw_words = [w for w in kw.split() if len(w) >= 3]
            exact   = (kw == tag)
            substr  = (len(kw) >= 4 and kw in tag) or (len(tag) >= 4 and tag in kw)
            word_hit = any((tw in kw) or any(kw_w in tw for kw_w in kw_words) for tw in tag_words)
            if exact or substr or word_hit:
                for pid in pids:
                    if pid not in product_map: continue
                    if kw in LOW_PRIORITY_TAGS:
                        matched_low.add(pid)
                    else:
                        score = conf * (1.0 if exact else 0.8 if substr else 0.6)
                        matched_high[pid] = matched_high.get(pid, 0) + score

    ranked   = sorted(matched_high.keys(), key=lambda p: matched_high[p], reverse=True)
    combined = list(ranked)
    if len(combined) < 4:
        for p in matched_low:
            if p not in matched_high:
                combined.append(p)
    return combined[:6]


def map_visual_label(raw_label: str) -> str:
    raw = raw_label.lower().strip()
    raw = re.sub(r"\(.*?\)", "", raw).strip()
    raw = raw.split(",")[0].strip().split("/")[0].strip().replace("_", " ").strip()
    if not raw or len(raw) < 3: return ""
    if raw in JUNK_DESCRIPTORS or raw in CUISINE_NOISE: return ""
    if raw in VISUAL_LABEL_TO_TAG: return VISUAL_LABEL_TO_TAG[raw]
    for key, val in VISUAL_LABEL_TO_TAG.items():
        if len(key) >= 5 and key in raw: return val
    for key, val in VISUAL_LABEL_TO_TAG.items():
        if len(raw) >= 5 and raw in key: return val
    first = raw.split()[0] if raw.split() else ""
    if len(first) >= 4 and first not in JUNK_DESCRIPTORS and first not in CUISINE_NOISE:
        if first in VISUAL_LABEL_TO_TAG: return VISUAL_LABEL_TO_TAG[first]
    return ""


# ─────────────────────────────────────────────
# OCR CLASSIFICATION (PRIORITY #1)
# ─────────────────────────────────────────────
def classify_image_ocr(image_bytes):
    """Extract text from image using pytesseract and match to grocery keywords.
    This runs FIRST before Gemini/HF to give priority to on-pack text."""
    debug = []
    try:
        import pytesseract
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Try multiple PSM modes — different modes work better for different layouts
        for psm in [6, 3, 11]:
            raw_text = pytesseract.image_to_string(image, config=f'--psm {psm}')
            text = re.sub(r'[^a-z0-9\s]', ' ', raw_text.lower())
            words = text.split()

            matched_tags = []
            seen_kw = set()

            # Check multi-word keywords first (higher specificity)
            for keyword in sorted(GROCERY_KEYWORDS.keys(), key=lambda k: -len(k)):
                kw = normalize_tag(keyword)
                if kw in seen_kw:
                    continue
                kw_words = kw.split()

                if len(kw_words) == 1:
                    # Single word: must appear as a full word in OCR output
                    if kw in words:
                        conf = 92 if len(kw) >= 5 else 80
                        matched_tags.append({"tag": kw, "confidence": conf})
                        seen_kw.add(kw)
                else:
                    # Multi-word: check if full phrase appears in text
                    if kw in text:
                        matched_tags.append({"tag": kw, "confidence": 95})
                        seen_kw.add(kw)

            if matched_tags:
                # Sort by confidence descending, deduplicate
                final = sorted(matched_tags, key=lambda x: x["confidence"], reverse=True)
                debug.append(f"✅ OCR (PSM {psm}) — matched: {[t['tag'] for t in final[:4]]}")
                return final[:6], None, debug

        debug.append("⚠️ OCR: no grocery keywords found in image text")
        return None, "no match", debug

    except ImportError:
        debug.append("⚠️ pytesseract not installed — skipping OCR step")
        return None, "no pytesseract", debug
    except Exception as ex:
        debug.append(f"⚠️ OCR error: {str(ex)[:80]}")
        return None, str(ex), debug


def classify_image_gemini(image_bytes):
    debug = []
    gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    if not gemini_key:
        debug.append("⚠️ GEMINI_API_KEY not set in secrets")
        return None, "no key", debug
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    # AQ. prefix = new Google AI Studio key format → use v1 endpoint
    api_ver = "v1" if gemini_key.startswith("AQ.") else "v1beta"
    for gmodel in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]:
        url = (f"https://generativelanguage.googleapis.com/{api_ver}/models/"
               f"{gmodel}:generateContent?key={gemini_key}")
        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                {"text": GEMINI_PROMPT}
            ]}],
            "generationConfig": {"temperature": 0.05, "topP": 0.8, "maxOutputTokens": 512}
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data  = resp.json()
                cands = data.get("candidates", [])
                if not cands:
                    debug.append(f"⚠️ {gmodel}: empty candidates"); continue
                fr = cands[0].get("finishReason", "")
                if fr in ("SAFETY", "RECITATION"):
                    debug.append(f"⚠️ {gmodel}: blocked ({fr})"); continue
                text = cands[0]["content"]["parts"][0]["text"].strip()
                text = text.replace("```json","").replace("```","").strip()
                s, e = text.find("["), text.rfind("]") + 1
                if s != -1 and e > s: text = text[s:e]
                result = json.loads(text)
                if result and isinstance(result, list):
                    cleaned = []
                    for item in result:
                        if isinstance(item, dict) and "tag" in item:
                            tag = normalize_tag(str(item["tag"]))
                            if tag in JUNK_DESCRIPTORS or tag in CUISINE_NOISE: continue
                            mapped = VISUAL_LABEL_TO_TAG.get(tag, tag)
                            if mapped and mapped not in JUNK_DESCRIPTORS and len(mapped) >= 3:
                                item["tag"] = mapped
                                cleaned.append(item)
                    if cleaned:
                        debug.append(f"✅ Gemini ({gmodel}) — {len(cleaned)} tags")
                        return cleaned, None, debug
            elif resp.status_code == 429:
                debug.append(f"⚠️ {gmodel}: rate limit"); continue
            elif resp.status_code == 400:
                err = resp.json().get("error", {}).get("message", "")
                debug.append(f"❌ {gmodel}: bad request — {err}"); break
            else:
                debug.append(f"❌ {gmodel}: HTTP {resp.status_code}"); break
        except json.JSONDecodeError as je:
            debug.append(f"⚠️ {gmodel}: JSON parse error — {je}"); continue
        except Exception as ex:
            debug.append(f"⚠️ {gmodel}: {str(ex)[:80]}"); continue
    return None, "Gemini failed", debug


def classify_image_hf(image_bytes):
    debug = []
    hf_token = str(st.secrets.get("HF_API_TOKEN", "")).strip()
    if not hf_token:
        debug.append("⚠️ HF_API_TOKEN not set in secrets")
        return None, "no key", debug
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "image/jpeg"}
    for model in ["nateraw/food", "Kaludi/grocery-products",
                  "google/vit-large-patch16-224", "microsoft/resnet-50"]:
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        try:
            resp = requests.post(url, headers=headers, data=image_bytes, timeout=30)
            if resp.status_code == 200:
                results = resp.json()
                if isinstance(results, list) and results:
                    tags, seen = [], set()
                    for item in results[:12]:
                        raw  = item.get("label","").lower().strip()
                        conf = round(item.get("score", 0.0) * 100, 1)
                        if conf < 4: continue
                        gtag = map_visual_label(raw)
                        if not gtag or len(gtag) < 3: continue
                        gtag = normalize_tag(gtag)
                        if gtag not in seen:
                            seen.add(gtag)
                            tags.append({"tag": gtag, "confidence": conf})
                    if tags:
                        debug.append(f"✅ HF ({model}) — {len(tags)} tags")
                        return tags, None, debug
                    else:
                        debug.append(f"⚠️ HF ({model}): no usable tags")
            elif resp.status_code in (503, 429):
                debug.append(f"⚠️ HF ({model}): {resp.status_code}"); continue
            else:
                debug.append(f"❌ HF ({model}): HTTP {resp.status_code}"); continue
        except Exception as ex:
            debug.append(f"⚠️ HF ({model}): {str(ex)[:60]}"); continue
    return None, "HF failed", debug


def color_fallback(image: Image.Image):
    img   = image.resize((100, 100)).convert("RGB")
    pix   = np.array(img).reshape(-1, 3).astype(float)
    mask  = ~((pix[:,0]>220) & (pix[:,1]>220) & (pix[:,2]>220))
    fg    = pix[mask] if mask.sum() > 50 else pix
    r, g, b  = fg.mean(axis=0)
    bright   = (r + g + b) / 3
    texture_std = fg.std(axis=0).mean()
    import colorsys
    hue, sat, val = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    hue_deg = hue * 360
    if 30 <= hue_deg <= 70 and sat > 0.25 and val > 0.45:
        if texture_std > 25:
            return [{"tag": "noodles", "confidence": 40}, {"tag": "chips", "confidence": 35},
                     {"tag": "biscuit", "confidence": 35}, {"tag": "namkeen", "confidence": 30}]
        if hue_deg >= 45 and texture_std < 12:
            return [{"tag": "banana", "confidence": 60}, {"tag": "juice", "confidence": 40}]
        if hue_deg < 45 and texture_std < 20:
            return [{"tag": "mango juice", "confidence": 55}, {"tag": "juice", "confidence": 50}]
        return [{"tag": "packaged food", "confidence": 40}, {"tag": "snack", "confidence": 35}]
    if g > r and g > b and g > 100 and g > r * 1.1:
        return [{"tag": "packaged food", "confidence": 55}]
    if r > g * 1.4 and r > b * 1.4 and r > 140 and hue_deg < 20:
        return [{"tag": "chilli powder", "confidence": 65}, {"tag": "masala", "confidence": 60}]
    if b > r * 1.1 and b > g * 1.1:
        return [{"tag": "milk", "confidence": 60}, {"tag": "dairy", "confidence": 55}]
    if bright > 210 and abs(r-g) < 20 and abs(g-b) < 20:
        return [{"tag": "butter", "confidence": 62}, {"tag": "dairy", "confidence": 55}]
    if bright < 80:
        return [{"tag": "coffee", "confidence": 60}, {"tag": "tea", "confidence": 55}]
    if r > 130 and g > 90 and b < 90 and r > g and r > b * 1.5:
        return [{"tag": "biscuit", "confidence": 60}]
    return [{"tag": "snack", "confidence": 52}, {"tag": "packaged food", "confidence": 50}]


# ─────────────────────────────────────────────
# MAIN CLASSIFY PIPELINE — OCR FIRST
# ─────────────────────────────────────────────
def classify_image(image_bytes):
    all_debug = []

    # ── STEP 1: OCR (highest priority — reads actual text on packaging)
    tags, err, dbg = classify_image_ocr(image_bytes)
    all_debug.extend(dbg)
    if tags:
        return tags, "📝 OCR Text Detection", all_debug

    # ── STEP 2: Gemini Vision API
    tags, err, dbg = classify_image_gemini(image_bytes)
    all_debug.extend(dbg)
    if tags:
        return tags, "✨ Gemini Vision", all_debug

    # ── STEP 3: HuggingFace Vision API
    tags, err, dbg = classify_image_hf(image_bytes)
    all_debug.extend(dbg)
    if tags:
        return tags, "🤗 HuggingFace Vision", all_debug

    # ── STEP 4: Color Fallback (always-on last resort)
    all_debug.append("⚠️ All methods failed — using color fallback")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tags  = color_fallback(image)
    return tags, "🎨 Color Fallback", all_debug


def get_cv_cf_recs(matched_pids, n=8):
    if not matched_pids: return []
    score_map   = {}
    matched_set = set(matched_pids)
    for base_pid in matched_pids[:3]:
        base_cat     = product_map.get(base_pid, {}).get('category', '')
        allowed_cats = RELATED_CATEGORIES.get(base_cat, [base_cat])
        if base_pid not in item_sim_df.index: continue
        sims = item_sim_df[base_pid].drop(base_pid).sort_values(ascending=False)
        sims = sims[[p for p in sims.index if product_map.get(p, {}).get('category') in allowed_cats]]
        for pid, score in sims.head(n*2).items():
            if pid not in matched_set:
                score_map[pid] = max(score_map.get(pid, 0), score)
    return sorted(score_map.keys(), key=lambda p: score_map[p], reverse=True)[:n]


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def stars(rating):
    r = float(rating) if rating else 0
    return '★' * int(r) + '☆' * (5 - int(r))

def product_card(pid, rank=None):
    p = product_map.get(pid, {})
    if not p: return
    name   = p.get('name', pid)
    cat    = p.get('category', '')
    sub    = p.get('subcategory', '')
    price  = p.get('price', '')
    rating = p.get('rating', '')
    emoji  = p.get('emoji', '🛒')
    tags   = p.get('tags', '')
    tag_html = ''.join(f'<span class="tag">#{t.strip()}</span>' for t in str(tags).split(',') if t.strip())
    rank_html = f'<span class="badge b-rank">#{rank}</span>' if rank else ''
    st.markdown(f"""
    <div class="prod-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.5rem">{emoji}</span>{rank_html}
        </div>
        <div class="prod-name" style="margin-top:6px">{name}</div>
        <div class="prod-meta">
            <span class="badge b-cat">{cat}</span>
            <span class="badge b-sub">{sub}</span>
            <span class="badge b-price">₹{price}</span>
            <span style="color:#f59e0b">{stars(rating)}</span>
            <span style="font-size:0.75rem;color:#475569"> {rating}/5</span>
        </div>
        <div style="margin-top:8px">{tag_html}</div>
    </div>
    """, unsafe_allow_html=True)

def product_card_cv(pid, score=None, score_label=""):
    p = product_map.get(pid, {})
    if not p: return
    name   = p.get('name', pid)
    cat    = p.get('category', '')
    price  = p.get('price', '')
    rating = p.get('rating', '')
    emoji  = p.get('emoji', '🛒')
    score_html = f'<div style="font-size:0.75rem;color:#34d399;margin-top:2px">{score_label}: {score:.3f}</div>' if score is not None else ''
    st.markdown(f"""
    <div class="prod-card-cv">
        <span style="font-size:2rem">{emoji}</span>
        <div class="prod-name-cv">{name}</div>
        <div style="color:#e74c3c;font-weight:700;font-size:0.85rem">₹{price}</div>
        <span class="badge b-green" style="margin-top:4px">{cat}</span>
        <div style="color:#f59e0b;font-size:0.75rem;margin-top:4px">{stars(rating)} {rating}/5</div>
        {score_html}
    </div>
    """, unsafe_allow_html=True)

def metric_card(value, label, hint="", color="#4f7ef8"):
    st.markdown(f"""
    <div class="eval-card">
        <div class="eval-val" style="color:{color}">{value}</div>
        <div class="eval-lbl">{label}</div>
        <div class="eval-hint">{hint}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Grocery AI")
    st.markdown("<p style='color:#475569;font-size:0.78rem;letter-spacing:0.05em;text-transform:uppercase'>Smart Recommender</p>", unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio("📌 Select Mode", [
        "🎯 User Recommendations",
        "🔍 Similar Products",
        "📸 Image Scanner",
        "🆕 Cold Start (New User)",
        "📊 Evaluation Metrics",
        "🔎 Search",
        "📋 Data Explorer"
    ])
    st.markdown("---")

    if mode == "🎯 User Recommendations":
        user_id    = st.selectbox("👤 Select User", all_users)
        algo       = st.selectbox("🤖 Algorithm", ["Hybrid (All 3)", "User-Based CF", "Item-Based CF", "SVD Matrix Factorization", "Popularity Based"])
        top_n      = st.slider("📦 Recommendations", 5, 20, 10)
        st.markdown("**🗂️ Category Filter** *(optional)*")
        cat_filter = st.multiselect("Filter categories", categories, default=[])
        cat_filter = cat_filter if cat_filter else None
        if algo == "Hybrid (All 3)":
            st.markdown("**⚖️ Algorithm Weights**")
            alpha = st.slider("User-Based weight", 0.1, 0.8, 0.4, 0.05)
            beta  = st.slider("Item-Based weight", 0.1, 0.8, 0.35, 0.05)
            if alpha + beta > 0.95:
                st.warning("Weights too high! SVD weight will be near 0.")
        else:
            alpha, beta = 0.4, 0.35
        run_btn = st.button("🚀 Get Recommendations", use_container_width=True)

    elif mode == "🔍 Similar Products":
        prod_labels = {pid: f"{product_map[pid]['emoji']} {product_map[pid]['name']}" for pid in all_pids}
        sel_pid     = st.selectbox("🧺 Select Product", all_pids, format_func=lambda x: prod_labels[x])
        sim_n       = st.slider("Similar items to show", 4, 16, 8)
        cat_filter  = st.multiselect("Filter categories", categories, default=[])
        cat_filter  = cat_filter if cat_filter else None
        sim_btn     = st.button("🔍 Find Similar Products", use_container_width=True)

    elif mode == "📸 Image Scanner":
        st.markdown("<p style='color:#8892b0;font-size:0.82rem'>Upload an image — AI identifies it and gives CF recommendations!</p>", unsafe_allow_html=True)
        n_cv_recs = st.slider("CF Suggestions", 4, 12, 8)

    elif mode == "🆕 Cold Start (New User)":
        st.markdown("<p style='color:#8892b0;font-size:0.82rem'>Select categories → get popularity-based recommendations!</p>", unsafe_allow_html=True)
        cs_cats  = st.multiselect("🗂️ Preferred Categories", categories, default=["Snacks", "Dairy"])
        cs_top_n = st.slider("Recommendations", 5, 20, 10)
        cs_btn   = st.button("✨ Get Recommendations", use_container_width=True)

    elif mode == "📊 Evaluation Metrics":
        eval_btn = st.button("📊 Compute Metrics", use_container_width=True)
        st.markdown("<p style='color:#8892b0;font-size:0.75rem'>80/20 train-test split. May take a few seconds.</p>", unsafe_allow_html=True)

    elif mode == "🔎 Search":
        search_type = st.radio("Search for", ["Products", "Users"])
        search_q    = st.text_input("🔎 Type to search...", placeholder="e.g. Amul, Snacks, U001")

    else:
        cat_exp = st.multiselect("Filter Category", categories, default=[])

    st.markdown("---")
    st.markdown("""
    <div style="color:#334155;font-size:0.72rem;text-align:center;letter-spacing:0.03em">
        500 Products · 150 Users · 6,796 Ratings<br>
        CF · SVD · CV Pipeline · Cold Start
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#e2e8f0;margin-bottom:2px">
    🛒 Smart Grocery Recommender
</h1>
<p style="color:#475569;font-size:0.92rem;margin-bottom:20px;letter-spacing:0.02em">
    Collaborative Filtering · Matrix Factorization · Computer Vision · Cold Start
</p>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col, num, lbl in zip([c1,c2,c3,c4,c5],
    ["500","150",f"{len(ratings):,}","13","5"],
    ["Products","Users","Ratings","Categories","Algorithms"]):
    with col:
        st.markdown(f'<div class="mbox"><div class="mnum">{num}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODE 1 — USER RECOMMENDATIONS
# ═══════════════════════════════════════════════
if mode == "🎯 User Recommendations":
    if run_btn:
        left, right = st.columns([3, 2])
        with left:
            algo_labels = {
                "Hybrid (All 3)": "🔀 Hybrid (UB + IB + SVD)",
                "User-Based CF": "👥 User-Based CF",
                "Item-Based CF": "📦 Item-Based CF",
                "SVD Matrix Factorization": "🧮 SVD Matrix Factorization",
                "Popularity Based": "🔥 Popularity Based"
            }
            st.markdown(f'<span class="badge b-algo">{algo_labels[algo]}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="sec-hdr">Recommendations for {user_id}</div>', unsafe_allow_html=True)
            with st.spinner("Computing recommendations..."):
                if algo == "Hybrid (All 3)":
                    recs = hybrid_recommend(user_id, top_n=top_n, alpha=alpha, beta=beta, cat_filter=cat_filter)
                elif algo == "User-Based CF":
                    recs = user_based_recommend(user_id, top_n=top_n, cat_filter=cat_filter)
                elif algo == "Item-Based CF":
                    recs = item_based_recommend(user_id, top_n=top_n, cat_filter=cat_filter)
                elif algo == "SVD Matrix Factorization":
                    recs = svd_recommend(user_id, top_n=top_n, cat_filter=cat_filter)
                else:
                    recs = popularity_recommend(top_n=top_n, cat_filter=cat_filter)
            if recs:
                for i, pid in enumerate(recs, 1):
                    product_card(pid, rank=i)
            else:
                st.warning("No recommendations found. Try removing category filters.")

        with right:
            st.markdown('<div class="sec-hdr">📜 Rating History</div>', unsafe_allow_html=True)
            _user_ratings = ratings[ratings['user_id'] == user_id].rename(columns={'rating': 'user_rating'})
            hist = _user_ratings.merge(products_df, on='product_id').sort_values('user_rating', ascending=False)

            st.markdown("**Category Preferences**")
            cat_counts = hist['category'].value_counts()
            for cat, cnt in cat_counts.items():
                pct = int(cnt / len(hist) * 100)
                w   = min(pct * 2.5, 100)
                st.markdown(f"""
                <div style="margin-bottom:6px">
                    <span style="font-size:0.78rem;color:#94a3b8;display:inline-block;width:110px">{cat}</span>
                    <div class="prog-wrap" style="display:inline-block;width:120px;vertical-align:middle">
                        <div class="prog-fill" style="width:{w}%"></div>
                    </div>
                    <span style="font-size:0.72rem;color:#64748b;margin-left:6px">{cnt}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Top Rated by This User**")
            for _, row in hist.head(8).iterrows():
                st.markdown(f"""
                <div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                    <span style="font-size:0.85rem;font-weight:500;color:#f1f5f9">{row.get('emoji','')} {row['name']}</span><br>
                    <span style="color:#f59e0b;font-size:0.78rem">{stars(row['user_rating'])}</span>
                    <span style="font-size:0.72rem;color:#475569"> {row['user_rating']} · {row['category']}</span>
                </div>""", unsafe_allow_html=True)
            avg = hist['user_rating'].mean()
            st.markdown(f"""
            <div style="margin-top:14px;background:rgba(139,92,246,0.08);border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(139,92,246,0.2)">
                <div style="font-size:1.4rem;font-weight:700;color:#a78bfa">{avg:.2f}</div>
                <div style="font-size:0.75rem;color:#64748b">Avg Rating · {len(hist)} products rated</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Select a user and algorithm from the sidebar, then click **Get Recommendations**.")


# ═══════════════════════════════════════════════
# MODE 2 — SIMILAR PRODUCTS
# ═══════════════════════════════════════════════
elif mode == "🔍 Similar Products":
    if sim_btn:
        p = product_map.get(sel_pid, {})
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(139,92,246,0.25);border-radius:16px;padding:20px 24px;margin-bottom:20px;backdrop-filter:blur(10px)">
            <div style="font-size:2.2rem">{p.get('emoji','🛒')}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:#f1f5f9;margin-top:4px">{p.get('name','')}</div>
            <div style="margin-top:8px">
                <span class="badge b-cat">{p.get('category','')}</span>
                <span class="badge b-sub">{p.get('subcategory','')}</span>
                <span class="badge b-price">₹{p.get('price','')}</span>
                <span style="color:#f59e0b"> {stars(p.get('rating',0))}</span>
                <span style="font-size:0.78rem;color:#475569"> {p.get('rating','')}/5</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<span class="badge b-algo">📦 Item-Based Similarity</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-hdr">Products Similar to "{p.get("name","")}"</div>', unsafe_allow_html=True)
        sim_pids = similar_items_cf(sel_pid, top_n=sim_n, cat_filter=cat_filter)
        if sim_pids:
            cols = st.columns(2)
            for i, pid in enumerate(sim_pids):
                with cols[i % 2]:
                    product_card(pid, rank=i+1)
        else:
            st.warning("Not enough interaction data for this product.")
    else:
        st.info("👈 Select a product and click **Find Similar Products**.")


# ═══════════════════════════════════════════════
# MODE 3 — IMAGE SCANNER
# ═══════════════════════════════════════════════
elif mode == "📸 Image Scanner":
    st.markdown('<div class="sec-hdr-green">📸 Image Scanner</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(52,211,153,0.06);border-radius:12px;padding:12px 16px;margin-bottom:16px;border-left:4px solid #34d399">
        <b style="color:#34d399">ℹ️ How it works:</b>
        <span style="font-size:0.85rem;color:#94a3b8">
        Upload image → <b>OCR reads on-pack text (highest priority)</b> →
        Gemini Vision fallback → HuggingFace fallback → Color analysis fallback →
        Matched products + CF-based suggestions appear instantly.
        </span>
    </div>
    """, unsafe_allow_html=True)

    up_col, prev_col = st.columns([1, 1])
    with up_col:
        uploaded_file = st.file_uploader("📤 Upload a grocery product image", type=["jpg","jpeg","png","webp"])

    if uploaded_file:
        image     = Image.open(uploaded_file).convert("RGB")
        img_bytes = uploaded_file.getvalue()
        with prev_col:
            st.markdown('<div class="img-preview">', unsafe_allow_html=True)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        b1, b2 = st.columns([1, 1])
        with b1:
            analyze_btn = st.button("🔍 Analyze & Recommend", use_container_width=True)
        with b2:
            if st.button("🔄 Change Image", use_container_width=True):
                for k in ["cv_done","cv_tags","cv_pids","cv_method","cv_debug"]:
                    st.session_state.pop(k, None)
                st.rerun()

        st.markdown("---")
        st.markdown('<div class="sec-hdr-green">🔎 Manual Search Override</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            manual_q = st.text_input("AI got it wrong? Type manually (e.g. maggi, butter, chips)", key="manual_cv")
        with mc2:
            st.markdown("<br>", unsafe_allow_html=True)
            manual_btn = st.button("🔍 Search", use_container_width=True)

        if manual_btn and manual_q.strip():
            q = manual_q.strip().lower()
            manual_tags = [{"tag": normalize_tag(q), "confidence": 95}]
            for w in q.split():
                if len(w) >= 3:
                    manual_tags.append({"tag": w, "confidence": 75})
            matched = find_products_from_tags(manual_tags, product_map)
            st.session_state["cv_tags"]   = manual_tags
            st.session_state["cv_pids"]   = matched
            st.session_state["cv_method"] = f"🔎 Manual: '{manual_q}'"
            st.session_state["cv_debug"]  = []
            st.session_state["cv_done"]   = True

        if analyze_btn:
            pb = st.progress(0, text="🧠 Initializing AI Vision...")
            import time; time.sleep(0.2)
            pb.progress(20, text="📝 Reading text on packaging (OCR)...")
            tags, method, debug_log = classify_image(img_bytes)
            pb.progress(75, text="🔍 Matching products in catalog...")
            time.sleep(0.2)
            matched = find_products_from_tags(tags, product_map)
            pb.progress(100, text="✅ Done!")
            time.sleep(0.3); pb.empty()
            st.session_state["cv_tags"]   = tags
            st.session_state["cv_pids"]   = matched
            st.session_state["cv_method"] = method
            st.session_state["cv_debug"]  = debug_log
            st.session_state["cv_done"]   = True

    if st.session_state.get("cv_done"):
        method  = st.session_state.get("cv_method", "")
        tags    = st.session_state.get("cv_tags", [])
        matched = st.session_state.get("cv_pids", [])
        primary = tags[0].get("tag","") if tags and isinstance(tags[0], dict) else str(tags[0]) if tags else ""

        st.markdown("---")
        st.markdown(f"""<div class="cv-banner">
            <span style="font-size:1.4rem">🔎</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#2ECC71;margin-left:0.5rem">
                Detected: {primary.title() if primary else "Unknown"}
            </span>
            <span style="font-size:0.8rem;color:#475569;margin-left:1rem">via {method}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr-green">🏷️ Detected Labels</div>', unsafe_allow_html=True)
        tags_html = ""
        for item in tags[:10]:
            if isinstance(item, dict):
                tname    = item.get("tag","")
                conf     = item.get("confidence", 0)
                conf_int = min(int(conf), 100)
                tags_html += f"""<div style="margin:4px 0;">
                    <span class="badge b-green">{tname}</span>
                    <span class="badge b-conf">{conf:.0f}%</span>
                    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf_int}%"></div></div>
                </div>"""
            else:
                tags_html += f'<span class="badge b-green">{item}</span>'
        st.markdown(f'<div style="margin:0.5rem 0 1rem 0">{tags_html}</div>', unsafe_allow_html=True)

        if st.session_state.get("cv_debug"):
            with st.expander("🔧 API Debug Log"):
                for msg in st.session_state["cv_debug"]:
                    st.markdown(f"`{msg}`")

        st.markdown('<div class="sec-hdr-green">🛒 Matched Products in Catalog</div>', unsafe_allow_html=True)
        if not matched:
            st.warning(f"⚠️ Detected **'{primary}'** but no exact catalog match. Try manual search above.")
        else:
            m_cols = st.columns(min(len(matched), 3))
            for i, pid in enumerate(matched):
                with m_cols[i % 3]:
                    product_card_cv(pid)

        if matched:
            st.markdown("---")
            st.markdown('<div class="sec-hdr-green">🤖 You Might Also Like (CF-Powered)</div>', unsafe_allow_html=True)
            base_cat = product_map.get(matched[0], {}).get('category', '')
            st.markdown(f'<small style="color:#64748b">Based on detected category: <b>{base_cat}</b> · Powered by Item-Based Collaborative Filtering</small>', unsafe_allow_html=True)
            cf_pids = get_cv_cf_recs(matched, n=n_cv_recs)
            if not cf_pids:
                st.info("No CF suggestions found.")
            else:
                cf_cols = st.columns(4)
                for i, pid in enumerate(cf_pids):
                    if pid not in product_map: continue
                    sim_score = item_sim_df.loc[matched[0], pid] if matched[0] in item_sim_df.index and pid in item_sim_df.columns else 0
                    with cf_cols[i % 4]:
                        product_card_cv(pid, score=sim_score, score_label="🔗 Sim")

        st.markdown("""
        <div style="background:rgba(52,211,153,0.06);border-radius:12px;padding:14px 18px;margin-top:16px;border-left:4px solid #34d399">
            <b style="color:#34d399">💡 CV Pipeline:</b>
            <span style="font-size:0.85rem;color:#94a3b8">
            Image → <b>OCR text extraction (priority)</b> →
            Gemini Vision API (fallback) →
            HuggingFace Inference API (fallback) →
            Color analysis (always-on fallback).
            Tags → GROCERY_KEYWORDS → product_ids → Item-CF similarity for "You Might Also Like".
            </span>
        </div>""", unsafe_allow_html=True)
    elif not uploaded_file:
        st.info("👈 **Image Scanner** is selected. Upload an image above to get started.")


# ═══════════════════════════════════════════════
# MODE 4 — COLD START
# ═══════════════════════════════════════════════
elif mode == "🆕 Cold Start (New User)":
    st.markdown("""
    <div class="cs-card">
        <h3 style="color:white;margin:0;font-family:'Space Grotesk',sans-serif">🆕 New User? No Problem!</h3>
        <p style="color:rgba(255,255,255,0.8);margin-top:8px;font-size:0.88rem">
            No purchase history? No problem. We use <b>Popularity-Based recommendations</b>
            from your preferred categories — the classic Cold Start solution.
        </p>
    </div>""", unsafe_allow_html=True)
    if cs_btn:
        if not cs_cats:
            st.warning("Please select at least one category!")
        else:
            st.markdown(f'<span class="badge b-algo">🔥 Popularity-Based (Cold Start)</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="sec-hdr">Top Picks in: {", ".join(cs_cats)}</div>', unsafe_allow_html=True)
            recs = cold_start_recommend(cs_cats, top_n=cs_top_n)
            if recs:
                cols = st.columns(2)
                for i, pid in enumerate(recs):
                    with cols[i % 2]:
                        product_card(pid, rank=i+1)
            else:
                st.warning("No products found in selected categories.")
            st.markdown("""
            <div style="background:rgba(251,191,36,0.06);border-radius:12px;padding:14px 18px;margin-top:16px;border-left:4px solid #f59e0b">
                <b style="color:#fdba74">💡 Cold Start:</b>
                <span style="font-size:0.85rem;color:#94a3b8"> Cold start occurs when a new user has no purchase history.
                Solutions: (1) Popularity-based fallback, (2) Content-based filtering, (3) Onboarding preference survey.</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Select categories in the sidebar and click **Get Recommendations**.")


# ═══════════════════════════════════════════════
# MODE 5 — EVALUATION METRICS
# ═══════════════════════════════════════════════
elif mode == "📊 Evaluation Metrics":
    st.markdown('<div class="sec-hdr">📊 Model Evaluation — 80/20 Train-Test Split</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(99,102,241,0.08);border-radius:12px;padding:14px 18px;margin-bottom:20px;border-left:4px solid #6366f1">
        <b style="color:#93c5fd">ℹ️ Methodology:</b>
        <span style="font-size:0.85rem;color:#94a3b8"> Ratings split into 80% train / 20% test.
        Models trained on train set, evaluated on test set. Threshold ≥ 3.5 = relevant item.</span>
    </div>""", unsafe_allow_html=True)
    if eval_btn:
        with st.spinner("⏳ Computing metrics... (thoda time lagega)"):
            metrics = compute_eval_metrics()
        K = metrics['K']
        st.markdown("### 📉 Error Metrics (Lower is Better)")
        c1, c2 = st.columns(2)
        with c1:
            metric_card(f"{metrics['rmse_svd']:.4f}", "RMSE — SVD Matrix Factorization",
                        "Root Mean Squared Error between predicted & actual ratings", color="#7b1fa2")
        with c2:
            metric_card(f"{metrics['rmse_ub']:.4f}", "RMSE — User-Based CF",
                        "Lower RMSE = predictions closer to actual ratings", color="#1565c0")
        st.markdown(f"### 🎯 Ranking Metrics @ K={K} (Higher is Better)")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(f"{metrics['precision_at_k']:.4f}", f"Precision@{K}",
                        f"What % of top-{K} recommendations were relevant", color="#2e7d32")
        with c2:
            metric_card(f"{metrics['recall_at_k']:.4f}", f"Recall@{K}",
                        "How many relevant items appeared in top-K", color="#e65100")
        with c3:
            metric_card(f"{metrics['f1']:.4f}", "F1 Score",
                        "Harmonic mean of Precision and Recall", color="#6a1b9a")
        st.markdown("### 🌐 Coverage (Higher is Better)")
        cov_pct = metrics['coverage'] * 100
        st.markdown(f"""
        <div class="eval-card">
            <div class="eval-val" style="color:#34d399">{cov_pct:.1f}%</div>
            <div class="eval-lbl">Catalog Coverage</div>
            <div class="eval-hint">% of catalog covered by recommendations (50-user sample)</div>
            <div class="prog-wrap" style="margin-top:10px">
                <div class="prog-fill" style="width:{min(cov_pct,100)}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 📋 Metrics Summary Table")
        summary = pd.DataFrame({
            'Metric':    [f'RMSE (SVD)', f'RMSE (User-CF)', f'Precision@{K}', f'Recall@{K}', 'F1 Score', 'Catalog Coverage'],
            'Value':     [f"{metrics['rmse_svd']:.4f}", f"{metrics['rmse_ub']:.4f}",
                          f"{metrics['precision_at_k']:.4f}", f"{metrics['recall_at_k']:.4f}",
                          f"{metrics['f1']:.4f}", f"{cov_pct:.1f}%"],
            'Direction': ['Lower ↓','Lower ↓','Higher ↑','Higher ↑','Higher ↑','Higher ↑'],
            'Description': [
                'SVD predicted vs actual rating error',
                'User-CF predicted vs actual rating error',
                f'Relevant items in top {K} recommendations',
                f'Relevant items retrieved in top {K}',
                'Balance of Precision & Recall',
                'Products covered by recommendations'
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("""
        <div style="background:rgba(52,211,153,0.06);border-radius:12px;padding:14px 18px;margin-top:8px;border-left:4px solid #34d399">
            <b style="color:#34d399">✅ Note:</b>
            <span style="font-size:0.85rem;color:#94a3b8"> Higher SVD RMSE is expected on synthetic/sparse data — real-world data typically yields 0.8–1.2.
            Precision@K and Coverage are the most industry-relevant metrics.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Click **Compute Metrics** in the sidebar.")


# ═══════════════════════════════════════════════
# MODE 6 — SEARCH
# ═══════════════════════════════════════════════
elif mode == "🔎 Search":
    st.markdown('<div class="sec-hdr">🔎 Search Products & Users</div>', unsafe_allow_html=True)
    if search_type == "Products" and search_q:
        q = search_q.strip().lower()
        results = products_df[
            products_df['name'].str.lower().str.contains(q, na=False) |
            products_df['category'].str.lower().str.contains(q, na=False) |
            products_df['subcategory'].str.lower().str.contains(q, na=False) |
            products_df['tags'].str.lower().str.contains(q, na=False)
        ]
        st.markdown(f"**{len(results)} results** for `{search_q}`")
        if len(results) == 0:
            st.warning("No products found. Try a different keyword.")
        else:
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.head(20).iterrows()):
                with cols[i % 2]:
                    product_card(row['product_id'])
    elif search_type == "Users" and search_q:
        q = search_q.strip().upper()
        matched_users = [u for u in all_users if q in u]
        st.markdown(f"**{len(matched_users)} users** found")
        if not matched_users:
            st.warning("No users found.")
        else:
            for uid in matched_users[:20]:
                hist = ratings[ratings['user_id'] == uid]
                avg  = hist['rating'].mean()
                merged = hist.merge(products_df[['product_id','category']], on='product_id')
                top_cat = merged['category'].value_counts().idxmax() if len(merged) else "N/A"
                st.markdown(f"""
                <div class="search-hit">
                    <b style="font-family:'Space Grotesk',sans-serif;color:#f1f5f9">{uid}</b>
                    <span style="font-size:0.78rem;color:#64748b;margin-left:10px">{len(hist)} ratings · Avg {avg:.2f} · Top: {top_cat}</span>
                </div>""", unsafe_allow_html=True)
    elif not search_q:
        st.info("👆 Type a product name, category, tag, or user ID to search.")


# ═══════════════════════════════════════════════
# MODE 7 — DATA EXPLORER
# ═══════════════════════════════════════════════
elif mode == "📋 Data Explorer":
    st.markdown('<div class="sec-hdr">📋 Dataset Explorer</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🛍️ Products", "⭐ Ratings", "📈 Insights"])
    with tab1:
        disp = products_df if not cat_exp else products_df[products_df['category'].isin(cat_exp)]
        st.markdown(f"Showing **{len(disp)}** products")
        st.dataframe(
            disp[['product_id','emoji','name','category','subcategory','price','rating','tags']],
            use_container_width=True, height=500, hide_index=True
        )
    with tab2:
        user_sel = st.selectbox("User", ["All"] + all_users)
        disp_r   = ratings if user_sel == "All" else ratings[ratings['user_id'] == user_sel]
        disp_r2  = disp_r.merge(products_df[['product_id','name','category','emoji']], on='product_id', how='left')
        st.markdown(f"Showing **{len(disp_r2)}** ratings")
        st.dataframe(
            disp_r2[['user_id','product_id','emoji','name','category','rating']],
            use_container_width=True, height=500, hide_index=True
        )
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ratings Count per Category**")
            rc = ratings.merge(products_df[['product_id','category']], on='product_id')['category'].value_counts()
            st.bar_chart(rc)
        with c2:
            st.markdown("**Rating Distribution**")
            rd = ratings['rating'].round(0).value_counts().sort_index()
            st.bar_chart(rd)
        st.markdown("**Top 15 Most Rated Products**")
        top_p = ratings.groupby('product_id').agg(
            ratings_count=('rating','count'), avg_rating=('rating','mean')
        ).sort_values('ratings_count', ascending=False).head(15).reset_index()
        top_p = top_p.merge(products_df[['product_id','name','category','emoji']], on='product_id')
        top_p['avg_rating'] = top_p['avg_rating'].round(2)
        st.dataframe(top_p[['emoji','name','category','ratings_count','avg_rating']], use_container_width=True, hide_index=True)
        st.markdown("**Ratings per User (Distribution)**")
        rpu = ratings.groupby('user_id').size().value_counts().sort_index()
        st.bar_chart(rpu)
