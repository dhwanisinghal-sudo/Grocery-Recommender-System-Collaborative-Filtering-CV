import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")

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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2ECC71, #27AE60, #1ABC9C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0;
        font-weight: 300;
        letter-spacing: 0.03em;
    }

    .card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }

    .card:hover {
        transform: translateY(-3px);
    }

    .product-card {
        background: linear-gradient(145deg, #ffffff, #f8fffe);
        border: 2px solid #d5f5e3;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46,204,113,0.1);
        transition: all 0.3s;
    }

    .product-card:hover {
        border-color: #2ECC71;
        box-shadow: 0 8px 25px rgba(46,204,113,0.25);
        transform: translateY(-2px);
    }

    .product-emoji {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    .product-name {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #1a1a2e;
    }

    .product-score {
        font-size: 0.8rem;
        color: #2ECC71;
        font-weight: 500;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-green { background: #d5f5e3; color: #1e8449; }
    .badge-blue  { background: #d6eaf8; color: #1a5276; }
    .badge-orange{ background: #fdebd0; color: #9c4e0c; }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #2ECC71;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #27AE60, #1e8449);
        box-shadow: 0 5px 20px rgba(46,204,113,0.4);
        transform: translateY(-2px);
    }

    .stat-box {
        background: linear-gradient(135deg, #2ECC71, #1ABC9C);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
    }

    .stat-number {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }

    .stat-label {
        font-size: 0.8rem;
        opacity: 0.85;
    }

    .detected-item {
        background: linear-gradient(135deg, #eafaf1, #d5f5e3);
        border: 2px solid #2ECC71;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #2ECC71 !important;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)

    products = {
        "P001": {"name": "Amul Full Cream Milk", "category": "Dairy", "emoji": "🥛", "price": 68, "tags": ["dairy", "protein", "breakfast"]},
        "P002": {"name": "Britannia Bread",      "category": "Bakery", "emoji": "🍞", "price": 45, "tags": ["bakery", "breakfast", "carbs"]},
        "P003": {"name": "Tata Salt",             "category": "Spices", "emoji": "🧂", "price": 25, "tags": ["spice", "essential", "cooking"]},
        "P004": {"name": "Nestlé Maggi",          "category": "Noodles","emoji": "🍜", "price": 14, "tags": ["snack", "quick", "kids"]},
        "P005": {"name": "Fortune Sunflower Oil", "category": "Oils",   "emoji": "🫙", "price": 145,"tags": ["oil", "cooking", "essential"]},
        "P006": {"name": "Lay's Classic Chips",   "category": "Snacks", "emoji": "🥔", "price": 20, "tags": ["snack", "kids", "party"]},
        "P007": {"name": "Aashirvaad Atta",       "category": "Grains", "emoji": "🌾", "price": 320,"tags": ["grain", "essential", "cooking"]},
        "P008": {"name": "Tropicana Orange Juice","category": "Drinks", "emoji": "🍊", "price": 99, "tags": ["drink", "vitamin", "breakfast"]},
        "P009": {"name": "Dabur Honey",           "category": "Health", "emoji": "🍯", "price": 215,"tags": ["health", "sweet", "breakfast"]},
        "P010": {"name": "Mother Dairy Curd",     "category": "Dairy",  "emoji": "🥗", "price": 55, "tags": ["dairy", "protein", "probiotic"]},
        "P011": {"name": "ITC Sunfeast Biscuit",  "category": "Bakery", "emoji": "🍪", "price": 30, "tags": ["bakery", "snack", "kids"]},
        "P012": {"name": "Haldiram's Namkeen",    "category": "Snacks", "emoji": "🥜", "price": 50, "tags": ["snack", "Indian", "party"]},
        "P013": {"name": "Parle-G Biscuit",       "category": "Bakery", "emoji": "🍪", "price": 10, "tags": ["bakery", "tea", "classic"]},
        "P014": {"name": "Kissan Mixed Fruit Jam","category": "Condiment","emoji":"🍓","price": 110,"tags": ["sweet", "breakfast", "kids"]},
        "P015": {"name": "Colgate Toothpaste",    "category": "Personal","emoji":"🪥", "price": 89, "tags": ["hygiene", "daily", "essential"]},
        "P016": {"name": "Ariel Detergent",       "category": "Home",   "emoji": "🧼", "price": 215,"tags": ["cleaning", "home", "essential"]},
        "P017": {"name": "Real Mango Juice",      "category": "Drinks", "emoji": "🥭", "price": 90, "tags": ["drink", "fruit", "summer"]},
        "P018": {"name": "Epigamia Greek Yogurt", "category": "Dairy",  "emoji": "🍦", "price": 75, "tags": ["dairy", "protein", "healthy"]},
        "P019": {"name": "Good Day Cashew Biscuit","category":"Bakery", "emoji": "🍘", "price": 35, "tags": ["bakery", "premium", "snack"]},
        "P020": {"name": "Green Tea Lipton",      "category": "Drinks", "emoji": "🍵", "price": 130,"tags": ["drink", "health", "morning"]},
    }

    users = [f"U{str(i).zfill(3)}" for i in range(1, 51)]
    product_ids = list(products.keys())

    # Simulate purchase matrix (0-5 rating, 0 = not bought)
    matrix = np.random.choice(
        [0, 0, 0, 1, 2, 3, 4, 5],
        size=(len(users), len(product_ids)),
        p=[0.5, 0.1, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02]
    )
    df = pd.DataFrame(matrix, index=users, columns=product_ids)
    return products, df


# ─────────────────────────────────────────────
# COLLABORATIVE FILTERING MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    # Matrix factorization via Truncated SVD
    svd = TruncatedSVD(n_components=10, random_state=42)
    user_factors = svd.fit_transform(df.values)
    item_factors = svd.components_.T

    # Reconstruct predicted ratings
    predicted = np.dot(user_factors, svd.components_)
    predicted_df = pd.DataFrame(predicted, index=df.index, columns=df.columns)

    # Item-item cosine similarity
    item_sim = cosine_similarity(item_factors)
    item_sim_df = pd.DataFrame(item_sim, index=df.columns, columns=df.columns)

    return predicted_df, item_sim_df


def get_user_recommendations(user_id, df, predicted_df, products, n=6):
    already_bought = df.loc[user_id][df.loc[user_id] > 0].index.tolist()
    preds = predicted_df.loc[user_id].copy()
    preds[already_bought] = -999  # Exclude already bought
    top_items = preds.nlargest(n).index.tolist()
    return top_items, already_bought


def get_similar_products(product_id, item_sim_df, n=5):
    if product_id not in item_sim_df.columns:
        return []
    sims = item_sim_df[product_id].drop(product_id).nlargest(n)
    return list(sims.index), list(sims.values)


# ─────────────────────────────────────────────
# CV: IMAGE CLASSIFICATION (via Clarifai/free API or fallback)
# ─────────────────────────────────────────────

GROCERY_KEYWORDS = {
    "milk": ["P001", "P010", "P018"],
    "bread": ["P002", "P013", "P019"],
    "biscuit": ["P011", "P013", "P019"],
    "cookie": ["P011", "P013", "P019"],
    "juice": ["P008", "P017"],
    "fruit": ["P008", "P017", "P014"],
    "oil": ["P005"],
    "salt": ["P003"],
    "noodle": ["P004"],
    "pasta": ["P004"],
    "chips": ["P006", "P012"],
    "snack": ["P006", "P012", "P011"],
    "flour": ["P007"],
    "wheat": ["P007"],
    "honey": ["P009"],
    "yogurt": ["P010", "P018"],
    "curd": ["P010", "P018"],
    "jam": ["P014"],
    "tea": ["P020"],
    "detergent": ["P016"],
    "soap": ["P016", "P015"],
    "toothpaste": ["P015"],
    "mango": ["P017"],
    "orange": ["P008"],
    "cashew": ["P019"],
    "namkeen": ["P012"],
    "atta": ["P007"],
    "maggi": ["P004"],
}

def classify_image_with_imagga(image_bytes):
    """Uses Imagga free API for image tagging"""
    try:
        api_key = st.secrets.get("IMAGGA_API_KEY", "")
        api_secret = st.secrets.get("IMAGGA_API_SECRET", "")

        if not api_key:
            return None, "No API key"

        b64 = base64.b64encode(image_bytes).decode()
        response = requests.post(
            "https://api.imagga.com/v2/tags",
            auth=(api_key, api_secret),
            data={"image_base64": b64},
            timeout=15
        )
        if response.status_code == 200:
            tags = response.json()["result"]["tags"]
            return [t["tag"]["en"].lower() for t in tags[:15]], None
        return None, f"API Error {response.status_code}"
    except Exception as e:
        return None, str(e)


def mock_classify_image(image: Image.Image):
    """Fallback: analyze image colors/properties as demo"""
    img_array = np.array(image.resize((50, 50)))
    avg_color = img_array.mean(axis=(0, 1))
    r, g, b = avg_color[0], avg_color[1], avg_color[2]

    # Color-based heuristic demo mapping
    if g > r and g > b and g > 100:
        labels = ["vegetable", "fresh produce", "green grocery"]
        matched = ["P007", "P003", "P005"]
    elif r > g and r > b and r > 150:
        labels = ["fruit", "tomato", "red food"]
        matched = ["P008", "P014", "P017"]
    elif b > r and b > g:
        labels = ["packaged drink", "bottle", "juice"]
        matched = ["P008", "P017", "P020"]
    elif abs(r-g) < 30 and abs(g-b) < 30 and r > 180:
        labels = ["dairy", "milk", "white product"]
        matched = ["P001", "P010", "P018"]
    elif r > 150 and g > 120 and b < 100:
        labels = ["snack", "chips", "packaged food"]
        matched = ["P006", "P012", "P011"]
    else:
        labels = ["grocery item", "food product", "packaged goods"]
        matched = ["P002", "P004", "P013"]

    return labels, matched


def find_products_from_tags(tags, products):
    matched = set()
    for tag in tags:
        for keyword, pids in GROCERY_KEYWORDS.items():
            if keyword in tag or tag in keyword:
                matched.update(pids)
    return list(matched)[:6]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛒 GrocerAI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🏠 Home Dashboard", "🤖 CF Recommendations", "📸 Image Scanner", "📊 Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_recs = st.slider("Number of Recommendations", 3, 10, 6)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;opacity:0.5;text-align:center;">Built with ❤️ using Streamlit<br>ML + CV Domain Project</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
products, ratings_df = generate_data()
predicted_df, item_sim_df = train_model(ratings_df)

users = ratings_df.index.tolist()
product_ids = list(products.keys())


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">🛒 Smart Grocery Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Collaborative Filtering + Computer Vision — ML/CV Domain Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">50</div>
            <div class="stat-label">Users</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">20</div>
            <div class="stat-label">Products</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">SVD</div>
            <div class="stat-label">CF Model</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">CV</div>
            <div class="stat-label">Vision Module</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📦 Product Catalog</div>', unsafe_allow_html=True)

    cats = list(set(v["category"] for v in products.values()))
    sel_cat = st.multiselect("Filter by Category", cats, default=cats[:4])

    cols = st.columns(4)
    for i, (pid, pdata) in enumerate(products.items()):
        if pdata["category"] in sel_cat:
            with cols[i % 4]:
                badges = "".join([f'<span class="badge badge-green">{t}</span>' for t in pdata["tags"][:2]])
                st.markdown(f"""
                <div class="product-card">
                    <span class="product-emoji">{pdata['emoji']}</span>
                    <div class="product-name">{pdata['name']}</div>
                    <div style="color:#e74c3c;font-weight:700;margin:0.25rem 0;">₹{pdata['price']}</div>
                    <div>{badges}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: COLLABORATIVE FILTERING
# ─────────────────────────────────────────────
elif page == "🤖 CF Recommendations":
    st.markdown('<h1 class="main-title">🤖 Collaborative Filtering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">SVD-based Matrix Factorization • Cosine Similarity</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["👤 User-Based Recommendations", "🔗 Item Similarity"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="section-header">Enter User</div>', unsafe_allow_html=True)

            input_mode = st.radio("Mode", ["📋 Existing User", "✏️ Any Custom User ID"], horizontal=True)

            if input_mode == "📋 Existing User":
                sel_user = st.selectbox("Choose from existing users", users)
            else:
                sel_user = st.text_input(
                    "Type any User ID",
                    placeholder="e.g. U999, Alice, john123...",
                    help="Type any name or ID — new users get popular item recommendations!"
                ).strip()

            if st.button("🎯 Get Recommendations") and sel_user:
                st.session_state["cf_user"] = sel_user
                st.session_state["cf_done"] = True

        with col2:
            if st.session_state.get("cf_done"):
                u = st.session_state["cf_user"]
                is_new_user = u not in ratings_df.index

                if is_new_user:
                    # Cold-start: recommend most popular items
                    st.markdown(f'<div class="detected-item">🆕 <b>New User: {u}</b> — No purchase history found. Showing <b>trending popular picks</b>!</div>', unsafe_allow_html=True)
                    popularity = ratings_df.astype(bool).sum(axis=0).sort_values(ascending=False)
                    recs = popularity.head(n_recs).index.tolist()
                    bought = []

                    st.markdown(f'<div class="section-header">🔥 Trending Recommendations for {u}</div>', unsafe_allow_html=True)
                    rcols = st.columns(3)
                    for i, pid in enumerate(recs):
                        p = products[pid]
                        buyers = int(popularity[pid])
                        with rcols[i % 3]:
                            st.markdown(f"""
                            <div class="product-card">
                                <span class="product-emoji">{p['emoji']}</span>
                                <div class="product-name">{p['name']}</div>
                                <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                                <div class="product-score">🔥 {buyers} buyers</div>
                            </div>""", unsafe_allow_html=True)
                else:
                    recs, bought = get_user_recommendations(u, ratings_df, predicted_df, products, n_recs)

                    st.markdown(f'<div class="section-header">✅ Already Purchased by {u}</div>', unsafe_allow_html=True)
                    bought_html = "".join([f'<span class="badge badge-blue">{products[p]["emoji"]} {products[p]["name"]}</span>' for p in bought[:8]])
                    st.markdown(f'<div style="margin-bottom:1rem;">{bought_html if bought_html else "<i style=\'color:#aaa\'>No purchases yet</i>"}</div>', unsafe_allow_html=True)

                    st.markdown(f'<div class="section-header">🎁 Recommended for {u}</div>', unsafe_allow_html=True)
                    rcols = st.columns(3)
                    for i, pid in enumerate(recs):
                        p = products[pid]
                        score = predicted_df.loc[u, pid]
                        with rcols[i % 3]:
                            st.markdown(f"""
                            <div class="product-card">
                                <span class="product-emoji">{p['emoji']}</span>
                                <div class="product-name">{p['name']}</div>
                                <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                                <div class="product-score">⭐ Score: {score:.2f}</div>
                            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">🔗 Item-Item Similarity</div>', unsafe_allow_html=True)
        sel_product = st.selectbox(
            "Select a product to find similar items",
            product_ids,
            format_func=lambda x: f"{products[x]['emoji']} {products[x]['name']}"
        )

        if st.button("🔍 Find Similar Products"):
            sim_pids, sim_scores = get_similar_products(sel_product, item_sim_df, n_recs)
            st.markdown(f'<div class="section-header">Products similar to {products[sel_product]["name"]}</div>', unsafe_allow_html=True)
            scols = st.columns(3)
            for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                p = products[pid]
                with scols[i % 3]:
                    st.markdown(f"""
                    <div class="product-card">
                        <span class="product-emoji">{p['emoji']}</span>
                        <div class="product-name">{p['name']}</div>
                        <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                        <div class="product-score">🔗 Similarity: {score:.3f}</div>
                    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: IMAGE SCANNER (CV)
# ─────────────────────────────────────────────
elif page == "📸 Image Scanner":
    st.markdown('<h1 class="main-title">📸 Product Image Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grocery item photo → CV identifies it → Recommends similar products</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("📌 **How it works:** Upload any grocery/food product image. The CV module analyzes it and maps it to products in our catalog, then uses Collaborative Filtering to suggest related items.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a grocery product image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo of any grocery item"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            img_bytes = uploaded_file.getvalue()

            if st.button("🔍 Analyze & Recommend"):
                with st.spinner("🧠 Running Computer Vision analysis..."):
                    # Try Imagga API first, fall back to color analysis
                    tags, err = classify_image_with_imagga(img_bytes)

                    if tags:
                        matched_pids = find_products_from_tags(tags, products)
                        method = "🌐 Imagga Vision API"
                    else:
                        tags, matched_pids = mock_classify_image(image)
                        method = "🎨 Color-based CV Analysis (Demo Mode)"

                    st.session_state["cv_tags"] = tags
                    st.session_state["cv_pids"] = matched_pids if matched_pids else ["P001", "P002", "P004"]
                    st.session_state["cv_method"] = method
                    st.session_state["cv_done"] = True

    with col2:
        if st.session_state.get("cv_done"):
            method = st.session_state["cv_method"]
            tags = st.session_state["cv_tags"]
            matched = st.session_state["cv_pids"]

            st.markdown(f'<div class="section-header">🏷️ Detected Labels</div>', unsafe_allow_html=True)
            st.markdown(f'<small style="color:#7f8c8d;">Method: {method}</small>', unsafe_allow_html=True)
            tags_html = "".join([f'<span class="badge badge-orange">{t}</span>' for t in tags[:10]])
            st.markdown(f'<div style="margin:0.75rem 0;">{tags_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">🛒 Matched Products</div>', unsafe_allow_html=True)
            if matched:
                for pid in matched:
                    p = products.get(pid, {})
                    if p:
                        st.markdown(f"""
                        <div class="detected-item">
                            <b>{p['emoji']} {p['name']}</b>
                            <span style="float:right;color:#2ECC71;font-weight:600;">₹{p['price']}</span><br>
                            <small style="color:#7f8c8d;">Category: {p['category']}</small>
                        </div>""", unsafe_allow_html=True)

                # CF-based extension: find similar to first matched product
                if matched:
                    st.markdown('<div class="section-header">🤖 CF-Enhanced Suggestions</div>', unsafe_allow_html=True)
                    sim_pids, sim_scores = get_similar_products(matched[0], item_sim_df, 4)
                    sc = st.columns(2)
                    for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                        p = products[pid]
                        with sc[i % 2]:
                            st.markdown(f"""
                            <div class="product-card">
                                <span class="product-emoji">{p['emoji']}</span>
                                <div class="product-name">{p['name']}</div>
                                <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                                <div class="product-score">🔗 {score:.3f}</div>
                            </div>""", unsafe_allow_html=True)
            else:
                st.warning("No matching products found. Try a different image.")

    st.markdown("---")
    st.markdown("### 🔧 Want Real CV Power?")
    st.markdown("""
    To enable production-grade image classification, add your **Imagga API** credentials in Streamlit secrets:
    ```toml
    # .streamlit/secrets.toml
    IMAGGA_API_KEY = "your_key_here"
    IMAGGA_API_SECRET = "your_secret_here"
    ```
    Get a **free API key** at [imagga.com](https://imagga.com) — 1000 free requests/month.
    """)


# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<h1 class="main-title">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Purchase patterns, model insights, and catalog stats</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🏆 Most Popular Products</div>', unsafe_allow_html=True)
        popularity = ratings_df.astype(bool).sum(axis=0).sort_values(ascending=False).head(10)
        pop_df = pd.DataFrame({
            "Product": [f"{products[p]['emoji']} {products[p]['name']}" for p in popularity.index],
            "Buyers": popularity.values
        })
        st.bar_chart(pop_df.set_index("Product"))

    with col2:
        st.markdown('<div class="section-header">📂 Category Distribution</div>', unsafe_allow_html=True)
        cat_counts = {}
        for p in products.values():
            cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
        cat_df = pd.DataFrame({"Category": list(cat_counts.keys()), "Count": list(cat_counts.values())})
        st.bar_chart(cat_df.set_index("Category"))

    st.markdown('<div class="section-header">📈 User Purchase Heatmap (Sample)</div>', unsafe_allow_html=True)
    sample = ratings_df.iloc[:15, :10]
    sample.columns = [f"{products[c]['emoji']}{products[c]['name'][:8]}" for c in sample.columns]
    st.dataframe(
        sample.style.background_gradient(cmap="Greens"),
        use_container_width=True
    )

    st.markdown('<div class="section-header">🔬 Model: SVD Explained Variance</div>', unsafe_allow_html=True)
    from sklearn.decomposition import TruncatedSVD
    svd_test = TruncatedSVD(n_components=10, random_state=42)
    svd_test.fit(ratings_df.values)
    var_df = pd.DataFrame({
        "Component": [f"C{i+1}" for i in range(10)],
        "Explained Variance (%)": (svd_test.explained_variance_ratio_ * 100).round(2)
    })
    st.bar_chart(var_df.set_index("Component"))
    st.caption(f"Total variance explained: {svd_test.explained_variance_ratio_.sum()*100:.1f}%")
