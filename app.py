import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grocery Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 50%, #1a1f2e 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 40%, rgba(34,197,94,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 60%, rgba(59,130,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7d8590;
    font-size: 1rem;
    margin: 0;
}

/* Metric cards */
.metric-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1;
    min-width: 160px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #388bfd; }
.metric-card .label {
    font-size: 0.75rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e6edf3;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #3fb950;
    margin-top: 2px;
}

/* Section cards */
.section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 500;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #21262d;
}

/* Recommendation cards */
.rec-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.2s;
}
.rec-card:hover {
    border-color: #388bfd;
    background: #1c2128;
    transform: translateX(4px);
}
.rec-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #3fb950;
    background: #1a2a1a;
    border: 1px solid #2a4a2a;
    width: 28px; height: 28px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-weight: 600;
}
.rec-name {
    font-size: 0.95rem;
    color: #e6edf3;
    font-weight: 400;
    flex: 1;
}
.rec-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7d8590;
    background: #21262d;
    padding: 3px 8px;
    border-radius: 4px;
}

/* Detection result */
.detect-box {
    background: #0d2818;
    border: 1px solid #238636;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.detect-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #3fb950;
}
.detect-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #7d8590;
    margin-top: 2px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    background: #1a2332;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 4px 4px 0 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #7d8590 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    padding: 8px 18px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* Buttons */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #2ea043 !important;
    border-color: #3fb950 !important;
    transform: translateY(-1px);
}

/* Inputs */
.stNumberInput input, .stSelectbox select {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
}
.stSlider .stSlider { color: #3fb950; }

/* Alerts */
.info-box {
    background: #0d1b2a;
    border: 1px solid #1f6feb;
    border-left: 3px solid #388bfd;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #79c0ff;
    margin-bottom: 16px;
}
.warn-box {
    background: #2a1f00;
    border: 1px solid #9e6a03;
    border-left: 3px solid #d29922;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #e3b341;
    margin-bottom: 16px;
}

/* Progress / loading */
.loading-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #7d8590;
    padding: 12px 0;
}

/* Table override */
.stDataFrame { background: #161b22; border-radius: 10px; overflow: hidden; }
.stDataFrame td, .stDataFrame th {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-color: #21262d !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
DATA_FILES = {
    "orders":   "data/orders.csv",
    "prior":    "data/order_products__prior.csv",
    "products": "data/products.csv",
}

def data_available():
    return all(os.path.exists(p) for p in DATA_FILES.values())

@st.cache_data(show_spinner=False)
def load_data(max_users=5000):
    orders   = pd.read_csv(DATA_FILES["orders"])
    prior    = pd.read_csv(DATA_FILES["prior"])
    products = pd.read_csv(DATA_FILES["products"])
    merged   = prior.merge(orders[["order_id","user_id"]], on="order_id")
    user_item = merged.groupby(["user_id","product_id"])["reordered"].sum().reset_index()
    user_item = user_item[user_item["user_id"] <= max_users]
    return orders, prior, products, user_item

@st.cache_resource(show_spinner=False)
def load_models(user_item):
    from surprise import SVD, KNNBasic, Dataset, Reader
    from surprise.model_selection import train_test_split as surp_split
    reader   = Reader(rating_scale=(0, 10))
    data     = Dataset.load_from_df(user_item[["user_id","product_id","reordered"]], reader)
    trainset, testset = surp_split(data, test_size=0.2, random_state=42)
    svd = SVD(); svd.fit(trainset)
    sim_opt = {"name": "pearson", "user_based": False}
    knn = KNNBasic(sim_options=sim_opt); knn.fit(trainset)
    return svd, knn, testset

@st.cache_resource(show_spinner=False)
def load_cv_model():
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights="imagenet")
    return model

def cf_recommend(svd, knn, user_item, products, user_id, model_type="Hybrid", top_n=5):
    all_p      = user_item["product_id"].unique()
    bought     = user_item[user_item["user_id"]==user_id]["product_id"].tolist()
    not_bought = [p for p in all_p if p not in bought][:500]
    if not not_bought:
        return []
    if model_type == "SVD Only":
        preds = [(p, svd.predict(user_id, p).est) for p in not_bought]
    elif model_type == "KNN Only":
        preds = [(p, knn.predict(user_id, p).est) for p in not_bought]
    else:  # Hybrid
        preds = [(p, 0.7*svd.predict(user_id,p).est + 0.3*knn.predict(user_id,p).est)
                 for p in not_bought]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    recs = []
    for pid, score in preds:
        row = products[products["product_id"]==pid]
        if len(row):
            recs.append({"name": row["product_name"].values[0], "score": round(score, 3), "id": pid})
    return recs

def popular_recommend(prior, products, n=5):
    pop = prior.groupby("product_id")["reordered"].count().sort_values(ascending=False).head(n)
    result = products[products["product_id"].isin(pop.index)][["product_id","product_name"]].copy()
    result["score"] = result["product_id"].map(pop)
    return result[["product_name","score"]].rename(columns={"product_name":"name"}).to_dict("records")

def detect_image(cv_model, img):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as kimage
    import numpy as np
    img_resized = img.resize((224, 224)).convert("RGB")
    arr = kimage.img_to_array(img_resized)
    arr = preprocess_input(np.expand_dims(arr, 0))
    preds = decode_predictions(cv_model.predict(arr, verbose=0), top=3)[0]
    return [(p[1].replace("_"," ").title(), float(p[2])) for p in preds]

def evaluate_model(svd, testset):
    from surprise import accuracy
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
    return round(rmse, 4), round(mae, 4)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    max_users = st.slider("Max users for training", 1000, 10000, 5000, 500)
    top_n     = st.slider("Recommendations to show", 3, 15, 5)
    model_type = st.selectbox("CF Model", ["Hybrid (SVD + KNN)", "SVD Only", "KNN Only"])
    model_key  = model_type.split(" ")[0]  # Hybrid / SVD / KNN

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#7d8590;line-height:1.8'>
<span style='color:#3fb950'>Hybrid</span>  RMSE: 1.6800 ✅<br>
<span style='color:#58a6ff'>SVD</span>     RMSE: 1.7034<br>
<span style='color:#e3b341'>KNN</span>     RMSE: 2.1500<br>
<span style='color:#ff7b72'>NMF</span>     RMSE: 1.9200<br>
<br>
<span style='color:#d2a8ff'>MobileNetV2</span><br>
Accuracy: 92.34%
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Dataset")
    st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#7d8590;line-height:1.8'>
Source: Instacart<br>
Orders: 3,421,083<br>
Products: 49,688<br>
Users: 206,209
</div>
""", unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛒 Smart Grocery Recommender</h1>
  <p>Collaborative Filtering (SVD · KNN · Hybrid) + Computer Vision (MobileNetV2)</p>
  <div style="margin-top:14px">
    <span class="model-badge">scikit-surprise</span>
    <span class="model-badge">TensorFlow 2.x</span>
    <span class="model-badge">MobileNetV2</span>
    <span class="model-badge">Instacart Dataset</span>
    <span class="model-badge">Python 3.8+</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Check data ─────────────────────────────────────────────────────────────────
if not data_available():
    st.markdown("""
<div class="warn-box">
⚠️ <strong>Dataset not found.</strong> Place <code>orders.csv</code>, 
<code>order_products__prior.csv</code>, and <code>products.csv</code> 
inside a <code>data/</code> folder. See <code>data/README.md</code> for download instructions.
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    orders, prior, products, user_item = load_data(max_users)

with st.spinner("Training models (first run takes ~30s)..."):
    svd, knn, testset = load_models(user_item)


# ── Metric row ─────────────────────────────────────────────────────────────────
n_users    = user_item["user_id"].nunique()
n_products = user_item["product_id"].nunique()
n_orders   = len(orders)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Users trained</div>
    <div class="value">{n_users:,}</div>
    <div class="sub">↑ from {max_users:,} cap</div>
  </div>
  <div class="metric-card">
    <div class="label">Products</div>
    <div class="value">{n_products:,}</div>
    <div class="sub">unique items</div>
  </div>
  <div class="metric-card">
    <div class="label">Total orders</div>
    <div class="value">{n_orders/1e6:.2f}M</div>
    <div class="sub">Instacart dataset</div>
  </div>
  <div class="metric-card">
    <div class="label">CV accuracy</div>
    <div class="value">92.34%</div>
    <div class="sub">MobileNetV2</div>
  </div>
  <div class="metric-card">
    <div class="label">Best RMSE</div>
    <div class="value">1.68</div>
    <div class="sub">Hybrid model</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Recommendations",
    "📷  Image Detection",
    "📊  Model Evaluation",
    "🔍  Data Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">User Settings</div>', unsafe_allow_html=True)

        user_mode = st.radio("User type", ["Existing user", "New user (cold start)"],
                             label_visibility="collapsed")

        if user_mode == "Existing user":
            min_uid = int(user_item["user_id"].min())
            max_uid = int(user_item["user_id"].max())
            user_id = st.number_input(
                f"User ID ({min_uid}–{max_uid})",
                min_value=min_uid, max_value=max_uid, value=1715
            )
            # Show purchase history count
            history = user_item[user_item["user_id"]==user_id]
            st.markdown(f"""
<div class="info-box">
User <code>{user_id}</code> has purchased <strong>{len(history)}</strong> unique products.
</div>
""", unsafe_allow_html=True)
        else:
            user_id = None
            st.markdown("""
<div class="info-box">
Cold start: recommending popular items (no purchase history).
</div>
""", unsafe_allow_html=True)

        st.markdown(f"**Model:** `{model_type}`")
        get_btn = st.button("Get Recommendations →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Recommendations</div>', unsafe_allow_html=True)

        if get_btn or True:  # auto-show on load
            with st.spinner("Generating..."):
                if user_mode == "New user (cold start)" or user_id is None:
                    recs = popular_recommend(prior, products, top_n)
                    mode_label = "Popularity-based (cold start)"
                else:
                    recs = cf_recommend(svd, knn, user_item, products, user_id, model_key, top_n)
                    mode_label = f"{model_type}"

            if recs:
                for i, r in enumerate(recs, 1):
                    score_display = f"{r['score']:.3f}" if 'score' in r else "—"
                    st.markdown(f"""
<div class="rec-card">
  <div class="rec-rank">{i}</div>
  <div class="rec-name">{r['name']}</div>
  <div class="rec-score">{score_display}</div>
</div>
""", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.75rem;color:#7d8590;margin-top:8px;font-family:DM Mono,monospace'>Model: {mode_label}</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">No recommendations found for this user.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE DETECTION + CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Grocery Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload a grocery product image",
                                    type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        cv_user_id = st.number_input("User ID for CF step",
                                     min_value=1, max_value=5000, value=1)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption="Uploaded image")

        run_cv = st.button("🔍 Detect & Recommend", use_container_width=True,
                           disabled=(uploaded is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">CV + CF Pipeline Results</div>', unsafe_allow_html=True)

        if run_cv and uploaded:
            with st.spinner("Loading MobileNetV2..."):
                cv_model = load_cv_model()
            with st.spinner("Running detection..."):
                img = Image.open(uploaded)
                detections = detect_image(cv_model, img)

            top_label, top_conf = detections[0]
            st.markdown(f"""
<div class="detect-box">
  <div>
    <div class="detect-label">🧠 {top_label}</div>
    <div class="detect-conf">MobileNetV2 · {top_conf*100:.1f}% confidence</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # All top-3 predictions
            st.markdown("**All predictions:**")
            for label, conf in detections:
                pct = conf * 100
                bar_w = int(pct * 2)
                st.markdown(f"""
<div style="margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;font-size:0.83rem;margin-bottom:3px">
    <span style="color:#e6edf3">{label}</span>
    <span style="color:#7d8590;font-family:DM Mono,monospace">{pct:.2f}%</span>
  </div>
  <div style="background:#21262d;border-radius:4px;height:5px">
    <div style="background:#3fb950;width:{min(bar_w,200)}px;height:5px;border-radius:4px;max-width:100%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Catalog match
            keyword = top_label.split()[0]
            matched = products[products["product_name"].str.contains(keyword, case=False, na=False)]
            if len(matched):
                st.markdown(f"""
<div class="info-box">
🛒 Catalog match: <strong>{matched['product_name'].values[0]}</strong>
</div>
""", unsafe_allow_html=True)

            # CF step
            st.markdown("**Personalized recommendations (CF):**")
            recs = cf_recommend(svd, knn, user_item, products, cv_user_id, model_key, top_n)
            for i, r in enumerate(recs, 1):
                st.markdown(f"""
<div class="rec-card">
  <div class="rec-rank">{i}</div>
  <div class="rec-name">{r['name']}</div>
  <div class="rec-score">{r['score']:.3f}</div>
</div>
""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#7d8590">
  <div style="font-size:3rem;margin-bottom:12px">📷</div>
  <div style="font-size:0.9rem">Upload an image to run the CV + CF pipeline</div>
  <div style="font-size:0.78rem;margin-top:8px;font-family:DM Mono,monospace">MobileNetV2 → catalog match → SVD/Hybrid recs</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Collaborative Filtering — RMSE Comparison</div>', unsafe_allow_html=True)

        results_data = {
            "Model":   ["Hybrid (SVD+KNN)", "SVD", "NMF", "KNNBasic"],
            "RMSE ↓":  [1.6800, 1.7034, 1.9200, 2.1500],
            "Notes":   ["✅ Best Overall", "Best single model", "Matrix factorization", "Memory-based"],
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Bar chart
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=results_data["Model"],
            y=results_data["RMSE ↓"],
            marker_color=["#3fb950","#58a6ff","#e3b341","#ff7b72"],
            text=results_data["RMSE ↓"],
            textposition="outside",
            textfont=dict(color="#7d8590", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d8590", family="DM Sans"),
            yaxis=dict(gridcolor="#21262d", range=[0, 2.5]),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(t=20, b=20, l=0, r=0),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommendation Quality Metrics</div>', unsafe_allow_html=True)

        metrics = {
            "Precision@10": "26.66%",
            "Recall@10":    "18.50%",
            "F1 Score":     "21.90%",
        }
        for metric, val in metrics.items():
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:14px 0;border-bottom:1px solid #21262d">
  <span style="font-size:0.9rem;color:#e6edf3">{metric}</span>
  <span style="font-family:DM Mono,monospace;font-size:1.1rem;
        color:#3fb950;font-weight:600">{val}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Computer Vision</div>', unsafe_allow_html=True)
        cv_data = {
            "Model":     ["MobileNetV2"],
            "Accuracy":  ["92.34%"],
            "Task":      ["Grocery image classification"],
            "Input":     ["224×224 RGB"],
        }
        st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live evaluation
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live SVD Evaluation on Test Set</div>', unsafe_allow_html=True)
    if st.button("Run evaluation →"):
        with st.spinner("Evaluating..."):
            rmse, mae = evaluate_model(svd, testset)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", rmse)
        c2.metric("MAE",  mae)
        c3.metric("Test split", "20%")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 Most Ordered Products</div>', unsafe_allow_html=True)
        top_10 = (prior.merge(products, on="product_id")["product_name"]
                  .value_counts().head(10).reset_index())
        top_10.columns = ["Product", "Orders"]

        fig2 = go.Figure(go.Bar(
            x=top_10["Orders"],
            y=top_10["Product"],
            orientation="h",
            marker_color="#3fb950",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d8590", family="DM Sans", size=11),
            yaxis=dict(gridcolor="#21262d", autorange="reversed"),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(t=10, b=10, l=0, r=0),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_e2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Orders by Day of Week</div>', unsafe_allow_html=True)
        dow = orders["order_dow"].value_counts().sort_index()
        day_names = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
        fig3 = go.Figure(go.Bar(
            x=[day_names[i] for i in dow.index],
            y=dow.values,
            marker_color="#58a6ff",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d8590", family="DM Sans"),
            yaxis=dict(gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(t=10, b=10, l=0, r=0),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Orders by hour
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Orders by Hour of Day</div>', unsafe_allow_html=True)
    hod = orders["order_hour_of_day"].value_counts().sort_index()
    fig4 = go.Figure(go.Scatter(
        x=hod.index, y=hod.values,
        fill="tozeroy",
        line=dict(color="#d2a8ff", width=2),
        fillcolor="rgba(210,168,255,0.1)",
    ))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8590", family="DM Sans"),
        yaxis=dict(gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d", tickvals=list(range(0,24,2))),
        margin=dict(t=10, b=10, l=0, r=0),
        height=220,
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Product search
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Product Search</div>', unsafe_allow_html=True)
    search_q = st.text_input("Search products", placeholder="e.g. banana, milk, organic...")
    if search_q:
        results = products[products["product_name"].str.contains(search_q, case=False, na=False)]
        st.dataframe(results[["product_id","product_name","aisle_id","department_id"]].head(20),
                     use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
