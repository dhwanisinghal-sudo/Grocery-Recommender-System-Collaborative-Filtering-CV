import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import requests
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
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title {
        font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #2ECC71, #27AE60, #1ABC9C);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;
    }
    .subtitle { font-size: 1rem; color: #7f8c8d; margin-top: 0; font-weight: 300; letter-spacing: 0.03em; }
    .card {
        background: linear-gradient(145deg, #1a1a2e, #16213e); border: 1px solid #0f3460;
        border-radius: 16px; padding: 1.5rem; margin: 0.75rem 0; color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3); transition: transform 0.2s;
    }
    .card:hover { transform: translateY(-3px); }
    .product-card {
        background: linear-gradient(145deg, #ffffff, #f8fffe); border: 2px solid #d5f5e3;
        border-radius: 12px; padding: 1rem; margin: 0.5rem 0; text-align: center;
        box-shadow: 0 4px 15px rgba(46,204,113,0.1); transition: all 0.3s;
    }
    .product-card:hover { border-color: #2ECC71; box-shadow: 0 8px 25px rgba(46,204,113,0.25); transform: translateY(-2px); }
    .product-emoji { font-size: 2.5rem; display: block; margin-bottom: 0.5rem; }
    .product-name { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.95rem; color: #1a1a2e; }
    .product-score { font-size: 0.8rem; color: #2ECC71; font-weight: 500; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.75rem; font-weight: 600; margin: 0.25rem; }
    .badge-green { background: #d5f5e3; color: #1e8449; }
    .badge-blue  { background: #d6eaf8; color: #1a5276; }
    .badge-orange{ background: #fdebd0; color: #9c4e0c; }
    .badge-conf  { background: #eaf4fb; color: #1a6ea8; font-size: 0.65rem; }
    .section-header {
        font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
        border-left: 4px solid #2ECC71; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2ECC71, #27AE60); color: white; border: none;
        border-radius: 10px; font-family: 'Syne', sans-serif; font-weight: 600;
        font-size: 0.95rem; padding: 0.6rem 1.5rem; transition: all 0.3s; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #27AE60, #1e8449);
        box-shadow: 0 5px 20px rgba(46,204,113,0.4); transform: translateY(-2px);
    }
    .stat-box { background: linear-gradient(135deg, #2ECC71, #1ABC9C); border-radius: 12px; padding: 1rem; text-align: center; color: white; }
    .stat-number { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.8rem; opacity: 0.85; }
    .detected-item {
        background: linear-gradient(135deg, #eafaf1, #d5f5e3); border: 2px solid #2ECC71;
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0;
    }
    .cart-item {
        background: #f8fffe; border: 1px solid #d5f5e3; border-radius: 10px;
        padding: 0.6rem 1rem; margin: 0.3rem 0; display: flex; justify-content: space-between;
        font-size: 0.85rem; color: #1a1a2e;
    }
    .cart-total {
        background: linear-gradient(135deg, #2ECC71, #27AE60); color: white;
        border-radius: 10px; padding: 0.75rem 1rem; text-align: center;
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; margin-top: 0.75rem;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    .sidebar-logo { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #2ECC71 !important; text-align: center; padding: 1rem 0; }
    .conf-bar-wrap { background: #e8f8f2; border-radius: 6px; height: 6px; margin: 3px 0 6px 0; }
    .conf-bar { background: #2ECC71; height: 6px; border-radius: 6px; }
    .image-preview-box {
        border: 2px dashed #2ECC71; border-radius: 12px; padding: 0.5rem;
        background: #f8fffe; text-align: center; margin-bottom: 0.75rem;
    }
    .image-preview-box img { max-height: 280px; border-radius: 8px; object-fit: contain; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CATEGORY → EMOJI MAPPING
# ─────────────────────────────────────────────
CATEGORY_EMOJI = {
    "Bakery": "🍪", "Snacks": "🥔", "Dairy": "🥛", "Grains": "🌾",
    "Spices": "🌶️", "Noodles": "🍜", "Drinks": "🥤", "Condiments": "🫙",
    "Personal Care": "🧴", "Health": "💊", "Home Care": "🧹",
    "Frozen": "🧊", "Beverages": "☕",
}

# RELATED CATEGORIES — for CF filter
# When a CV detects a product from category X, only suggest CF results from these categories
RELATED_CATEGORIES = {
    "Drinks":        ["Drinks", "Beverages", "Health"],
    "Beverages":     ["Beverages", "Drinks", "Health"],
    "Snacks":        ["Snacks", "Bakery", "Drinks", "Beverages"],
    "Bakery":        ["Bakery", "Snacks", "Dairy", "Condiments"],
    "Dairy":         ["Dairy", "Bakery", "Health", "Beverages"],
    "Grains":        ["Grains", "Spices", "Condiments", "Noodles"],
    "Spices":        ["Spices", "Grains", "Condiments", "Noodles"],
    "Noodles":       ["Noodles", "Grains", "Spices", "Condiments"],
    "Condiments":    ["Condiments", "Spices", "Grains", "Noodles"],
    "Personal Care": ["Personal Care", "Health", "Home Care"],
    "Health":        ["Health", "Dairy", "Beverages", "Personal Care"],
    "Home Care":     ["Home Care", "Personal Care"],
    "Frozen":        ["Frozen", "Snacks", "Dairy"],
}

def get_emoji(category):
    return CATEGORY_EMOJI.get(category, "🛒")


# ─────────────────────────────────────────────
# LOAD DATA FROM CSV
# ─────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_data():
    import os
    possible_paths = ["data/products.csv", "./data/products.csv", "products.csv"]
    products_df = None
    for path in possible_paths:
        if os.path.exists(path):
            products_df = pd.read_csv(path)
            break
    if products_df is None:
        st.error("❌ products.csv not found! Paths tried: " + str(possible_paths))
        st.write("📁 Current directory:", os.getcwd())
        st.write("📂 Files here:", os.listdir("."))
        st.stop()

    products = {}
    for _, row in products_df.iterrows():
        pid = row["product_id"]
        tags = [t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()]
        emoji = str(row.get("emoji", "")).strip()
        if not emoji or emoji == "nan":
            emoji = get_emoji(row["category"])
        products[pid] = {
            "name":     row["name"],
            "category": row["category"],
            "emoji":    emoji,
            "price":    int(row["price"]),
            "tags":     tags,
            "rating":   float(row.get("rating", 4.0)),
        }

    user_ids_existing = [f"U{str(i).zfill(3)}" for i in range(1, 51)]
    try:
        users_paths = ["data/users_new.csv", "./data/users_new.csv", "users_new.csv"]
        new_users_df = None
        for p in users_paths:
            if os.path.exists(p):
                new_users_df = pd.read_csv(p)
                break
        user_ids_new = new_users_df["user_id"].tolist() if new_users_df is not None else []
    except Exception:
        user_ids_new = []

    all_user_ids = user_ids_existing + user_ids_new
    product_ids  = list(products.keys())

    ratings_paths = ["data/ratings.csv", "./data/ratings.csv", "ratings.csv"]
    ratings_path = next((p for p in ratings_paths if os.path.exists(p)), None)
    try:
        ratings_raw = pd.read_csv(ratings_path) if ratings_path else None
        if ratings_raw is None: raise FileNotFoundError
        matrix = ratings_raw.pivot_table(
            index="user_id", columns="product_id", values="rating", aggfunc="mean"
        )
        matrix = matrix.reindex(index=all_user_ids, columns=product_ids, fill_value=0).fillna(0)
    except FileNotFoundError:
        np.random.seed(42)
        raw = np.random.choice(
            [0, 0, 0, 1, 2, 3, 4, 5],
            size=(len(all_user_ids), len(product_ids)),
            p=[0.5, 0.1, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02]
        )
        matrix = pd.DataFrame(raw, index=all_user_ids, columns=product_ids)

    return products, matrix


# ─────────────────────────────────────────────
# COLLABORATIVE FILTERING MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    n_components = min(20, _df.shape[0] - 1, _df.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(_df.values)
    item_factors = svd.components_.T
    predicted    = np.dot(user_factors, svd.components_)
    predicted_df = pd.DataFrame(predicted, index=_df.index, columns=_df.columns)
    item_sim     = cosine_similarity(item_factors)
    item_sim_df  = pd.DataFrame(item_sim, index=_df.columns, columns=_df.columns)
    return predicted_df, item_sim_df


def get_user_recommendations(user_id, df, predicted_df, n=6):
    already_bought = df.loc[user_id][df.loc[user_id] > 0].index.tolist()
    preds = predicted_df.loc[user_id].copy()
    preds[already_bought] = -999
    top_items = preds.nlargest(n).index.tolist()
    return top_items, already_bought


def get_similar_products(product_id, item_sim_df, products, n=5, filter_categories=None):
    """
    FIX: Added category filter so CF doesn't suggest irrelevant products.
    filter_categories — list of allowed categories. If None, no filter applied.
    """
    if product_id not in item_sim_df.columns:
        return [], []
    sims = item_sim_df[product_id].drop(product_id).sort_values(ascending=False)

    if filter_categories:
        # Only keep product IDs whose category is in allowed list
        allowed_pids = [pid for pid in sims.index if products.get(pid, {}).get("category") in filter_categories]
        sims = sims[allowed_pids]

    top = sims.head(n)
    return list(top.index), list(top.values)


# ─────────────────────────────────────────────
# CART HELPERS
# ─────────────────────────────────────────────
def init_cart():
    if "cart" not in st.session_state:
        st.session_state["cart"] = {}  # {pid: {"name": ..., "price": ..., "emoji": ..., "qty": ...}}

def add_to_cart(pid, products):
    init_cart()
    if pid in st.session_state["cart"]:
        st.session_state["cart"][pid]["qty"] += 1
    else:
        p = products[pid]
        st.session_state["cart"][pid] = {"name": p["name"], "price": p["price"], "emoji": p["emoji"], "qty": 1}

def render_cart_sidebar():
    init_cart()
    cart = st.session_state["cart"]
    st.markdown("---")
    st.markdown("### 🛒 Cart")
    if not cart:
        st.markdown('<p style="font-size:0.8rem;opacity:0.6;">Cart is empty</p>', unsafe_allow_html=True)
        return
    total = 0
    for pid, item in list(cart.items()):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f'<div style="font-size:0.8rem;color:#ecf0f1;">{item["emoji"]} {item["name"]} ×{item["qty"]}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div style="font-size:0.8rem;color:#2ECC71;">₹{item["price"]*item["qty"]}</div>', unsafe_allow_html=True)
        total += item["price"] * item["qty"]
    st.markdown(f'<div style="background:#2ECC71;border-radius:8px;padding:0.5rem;text-align:center;color:white;font-weight:700;margin-top:0.5rem;">Total: ₹{total}</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Cart"):
        st.session_state["cart"] = {}
        st.rerun()


# ─────────────────────────────────────────────
# CV: IMAGE CLASSIFICATION
# ─────────────────────────────────────────────
GROCERY_KEYWORDS = {
    "milk": ["P021","P030","P136"], "bread": ["P001","P003","P132"],
    "biscuit": ["P001","P002","P007"], "cookie": ["P006","P133","P134"],
    "juice": ["P061","P062","P063"], "fruit": ["P061","P062","P137"],
    "oil": ["P049","P050"], "salt": ["P003","P041"],
    "noodle": ["P051","P052","P057"], "pasta": ["P055","P056","P060"],
    "chips": ["P011","P012","P018"], "snack": ["P011","P015","P016"],
    "flour": ["P035","P036","P145"], "wheat": ["P035","P145"],
    "honey": ["P077","P096"], "yogurt": ["P023","P029","P030"],
    "curd": ["P023","P030"], "jam": ["P071","P080"],
    "tea": ["P121","P122","P125","P126"], "coffee": ["P123","P124","P127"],
    "detergent": ["P101","P102"], "soap": ["P083","P084"],
    "toothpaste": ["P081"], "mango": ["P062","P063","P066"],
    "orange": ["P061"], "cashew": ["P002"],
    "namkeen": ["P015","P016","P017"], "atta": ["P035","P145"],
    "maggi": ["P051","P131"], "rice": ["P031","P032"],
    "dal": ["P033","P034"], "ghee": ["P025"],
    "butter": ["P021"], "masala": ["P041","P042","P047"],
    "banana": ["P061","P062","P063","P137"],
    "apple": ["P061","P062","P137"],
    "vegetable": ["P041","P049","P033","P034"],
    "food": ["P001","P051","P061"],
    "yellow": ["P062","P063","P061"],
    "ripe": ["P061","P062","P137"],
}

# FIX 1: Confidence threshold — only accept tags above this % confidence
CONFIDENCE_THRESHOLD = 55.0

# Non-food / irrelevant keywords to ignore even if API returns them
IGNORE_KEYWORDS = {
    "sweet pepper", "pepper", "capsicum", "paprika", "chili",
    "spice", "seasoning", "herb", "ingredient",
}

def classify_image_with_imagga(image_bytes):
    """
    FIX: Returns (tags_with_confidence, error) where tags are filtered by
    CONFIDENCE_THRESHOLD and IGNORE_KEYWORDS.
    """
    try:
        api_key    = st.secrets.get("IMAGGA_API_KEY", "")
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
            raw_tags = response.json()["result"]["tags"]
            # Filter by confidence threshold + ignore list
            filtered = [
                {"tag": t["tag"]["en"].lower(), "confidence": round(t["confidence"], 1)}
                for t in raw_tags
                if t["confidence"] >= CONFIDENCE_THRESHOLD
                and t["tag"]["en"].lower() not in IGNORE_KEYWORDS
            ]
            return filtered[:12], None
        return None, f"API Error {response.status_code}"
    except Exception as e:
        return None, str(e)


def mock_classify_image(image: Image.Image):
    img_array = np.array(image.resize((50, 50)))
    avg_color = img_array.mean(axis=(0, 1))
    r, g, b   = avg_color[0], avg_color[1], avg_color[2]

    # Yellow = high R, high G, low B (banana, mango, lemon)
    is_yellow = r > 160 and g > 140 and b < 120 and abs(r - g) < 60
    # Green = G dominates
    is_green  = g > r and g > b and g > 100
    # Red = R dominates clearly
    is_red    = r > g + 30 and r > b + 30 and r > 150
    # White/grey = all channels similar and high
    is_white  = abs(r-g) < 30 and abs(g-b) < 30 and r > 180
    # Blue = B dominates
    is_blue   = b > r and b > g
    # Orange-brown (snacks) = high R, medium G, low B
    is_orange = r > 150 and g > 100 and g < 160 and b < 100 and not is_yellow

    if is_yellow:
        tags = [{"tag": "banana", "confidence": 91}, {"tag": "fruit", "confidence": 86},
                {"tag": "yellow", "confidence": 80}, {"tag": "ripe", "confidence": 73}]
        pids = ["P061","P062","P063"]
    elif is_green:
        tags = [{"tag": "vegetable", "confidence": 82}, {"tag": "fresh produce", "confidence": 74},
                {"tag": "green grocery", "confidence": 65}]
        pids = ["P035","P041","P049"]
    elif is_red:
        tags = [{"tag": "fruit", "confidence": 85}, {"tag": "tomato", "confidence": 77},
                {"tag": "red food", "confidence": 68}]
        pids = ["P061","P071","P062"]
    elif is_white:
        tags = [{"tag": "dairy", "confidence": 83}, {"tag": "milk", "confidence": 77},
                {"tag": "white product", "confidence": 70}]
        pids = ["P021","P023","P029"]
    elif is_blue:
        tags = [{"tag": "packaged drink", "confidence": 85}, {"tag": "bottle", "confidence": 76},
                {"tag": "juice", "confidence": 68}]
        pids = ["P061","P062","P125"]
    elif is_orange:
        tags = [{"tag": "snack", "confidence": 80}, {"tag": "chips", "confidence": 73},
                {"tag": "packaged food", "confidence": 65}]
        pids = ["P011","P015","P016"]
    else:
        tags = [{"tag": "grocery item", "confidence": 75}, {"tag": "food product", "confidence": 67},
                {"tag": "packaged goods", "confidence": 60}]
        pids = ["P001","P051","P121"]
    return tags, pids


def find_products_from_tags(tag_dicts, products):
    """Works with new tag format: list of {"tag": ..., "confidence": ...}"""
    matched = set()
    for item in tag_dicts:
        tag = item["tag"] if isinstance(item, dict) else item
        for keyword, pids in GROCERY_KEYWORDS.items():
            if keyword in tag or tag in keyword:
                for pid in pids:
                    if pid in products:
                        matched.add(pid)
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
    render_cart_sidebar()
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;opacity:0.5;text-align:center;">Built with ❤️ using Streamlit<br>ML + CV Domain Project</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL
# ─────────────────────────────────────────────
products, ratings_df = load_data()
predicted_df, item_sim_df = train_model(ratings_df)
init_cart()

users       = ratings_df.index.tolist()
product_ids = list(products.keys())
n_users     = len(users)
n_products  = len(products)


# ─────────────────────────────────────────────
# HELPER: Product Card with Add to Cart
# ─────────────────────────────────────────────
def render_product_card_with_cart(pid, p, score_label=None, col=None):
    """Renders a product card. col is the st.column context."""
    ctx = col if col else st
    tags_html = "".join([f'<span class="badge badge-green">{t}</span>' for t in p["tags"][:2]])
    score_html = f'<div class="product-score">{score_label}</div>' if score_label else ""
    ctx.markdown(f"""
    <div class="product-card">
        <span class="product-emoji">{p['emoji']}</span>
        <div class="product-name">{p['name']}</div>
        <div style="color:#e74c3c;font-weight:700;margin:0.3rem 0;">₹{p['price']}</div>
        <div style="margin-bottom:0.4rem;">{tags_html}</div>
        {score_html}
    </div>""", unsafe_allow_html=True)
    if ctx.button(f"🛒 Add to Cart", key=f"cart_{pid}_{id(score_label)}"):
        add_to_cart(pid, products)
        st.toast(f"✅ {p['name']} added to cart!", icon="🛒")


# ─────────────────────────────────────────────
# PAGE: HOME DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">🛒 Smart Grocery Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Collaborative Filtering + Computer Vision — ML/CV Domain Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="stat-box"><div class="stat-number">{n_users}</div><div class="stat-label">Users</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-box"><div class="stat-number">{n_products}</div><div class="stat-label">Products</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="stat-box"><div class="stat-number">SVD</div><div class="stat-label">CF Model</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="stat-box"><div class="stat-number">CV</div><div class="stat-label">Vision Module</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📦 Product Catalog</div>', unsafe_allow_html=True)

    cats     = sorted(set(v["category"] for v in products.values()))
    sel_cats = st.multiselect("Filter by Category", cats, default=cats[:4])

    filtered = {pid: pdata for pid, pdata in products.items() if pdata["category"] in sel_cats}

    cols = st.columns(4)
    for i, (pid, pdata) in enumerate(filtered.items()):
        with cols[i % 4]:
            badges = "".join([f'<span class="badge badge-green">{t}</span>' for t in pdata["tags"][:2]])
            st.markdown(f"""
            <div class="product-card">
                <span class="product-emoji">{pdata['emoji']}</span>
                <div class="product-name">{pdata['name']}</div>
                <div style="color:#e74c3c;font-weight:700;margin:0.25rem 0;">₹{pdata['price']}</div>
                <div>{badges}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("🛒 Add", key=f"home_cart_{pid}"):
                add_to_cart(pid, products)
                st.toast(f"✅ {pdata['name']} added!", icon="🛒")


# ─────────────────────────────────────────────
# PAGE: CF RECOMMENDATIONS
# ─────────────────────────────────────────────
elif page == "🤖 CF Recommendations":
    st.markdown('<h1 class="main-title">🤖 Collaborative Filtering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">SVD-based Matrix Factorization • Cosine Similarity</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["👤 User-Based Recommendations", "🔗 Item Similarity"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="section-header">Select User</div>', unsafe_allow_html=True)
            sel_user = st.selectbox("Choose a user", users)
            if st.button("🎯 Get Recommendations"):
                st.session_state["cf_user"] = sel_user
                st.session_state["cf_done"] = True

        with col2:
            if st.session_state.get("cf_done"):
                u = st.session_state["cf_user"]
                recs, bought = get_user_recommendations(u, ratings_df, predicted_df, n_recs)
                st.markdown(f'<div class="section-header">✅ Already Purchased by {u}</div>', unsafe_allow_html=True)
                bought_html = "".join([
                    f'<span class="badge badge-blue">{products[p]["emoji"]} {products[p]["name"]}</span>'
                    for p in bought[:8] if p in products
                ])
                st.markdown(f'<div style="margin-bottom:1rem;">{bought_html or "<i style=color:#aaa>No purchases yet</i>"}</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="section-header">🎁 Recommended for {u}</div>', unsafe_allow_html=True)
                rcols = st.columns(3)
                for i, pid in enumerate(recs):
                    if pid not in products:
                        continue
                    p     = products[pid]
                    score = predicted_df.loc[u, pid]
                    with rcols[i % 3]:
                        st.markdown(f"""
                        <div class="product-card">
                            <span class="product-emoji">{p['emoji']}</span>
                            <div class="product-name">{p['name']}</div>
                            <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                            <div class="product-score">⭐ Score: {score:.2f}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cf_rec_{pid}_{u}"):
                            add_to_cart(pid, products)
                            st.toast(f"✅ {p['name']} added!", icon="🛒")

    with tab2:
        st.markdown('<div class="section-header">🔗 Item-Item Similarity</div>', unsafe_allow_html=True)
        sel_product = st.selectbox(
            "Select a product",
            product_ids,
            format_func=lambda x: f"{products[x]['emoji']} {products[x]['name']}"
        )
        if st.button("🔍 Find Similar Products"):
            # FIX: Filter CF results to same/related category
            base_cat = products[sel_product]["category"]
            allowed  = RELATED_CATEGORIES.get(base_cat, [base_cat])
            sim_pids, sim_scores = get_similar_products(sel_product, item_sim_df, products, n_recs, filter_categories=allowed)
            st.markdown(f'<div class="section-header">Products similar to {products[sel_product]["name"]}</div>', unsafe_allow_html=True)
            if not sim_pids:
                st.info("No similar products found in related categories.")
            scols = st.columns(3)
            for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                if pid not in products:
                    continue
                p = products[pid]
                with scols[i % 3]:
                    st.markdown(f"""
                    <div class="product-card">
                        <span class="product-emoji">{p['emoji']}</span>
                        <div class="product-name">{p['name']}</div>
                        <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                        <div class="product-score">🔗 Similarity: {score:.3f}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("🛒 Add to Cart", key=f"sim_{pid}_{sel_product}"):
                        add_to_cart(pid, products)
                        st.toast(f"✅ {p['name']} added!", icon="🛒")


# ─────────────────────────────────────────────
# PAGE: IMAGE SCANNER
# ─────────────────────────────────────────────
elif page == "📸 Image Scanner":
    st.markdown('<h1 class="main-title">📸 Product Image Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grocery photo → CV identifies it → Recommends similar products</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("📌 Upload any grocery/food product image. The CV module analyzes it and maps it to products in our catalog, then uses CF to suggest related items.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a grocery product image", type=["jpg","jpeg","png","webp"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            # FIX: Proper bounded image preview instead of full-width
            st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
            st.image(image, caption="📷 Uploaded Image", width=300)
            st.markdown('</div>', unsafe_allow_html=True)

            # Change image button
            if st.button("🔄 Change Image"):
                st.session_state["cv_done"] = False
                st.rerun()

            img_bytes = uploaded_file.getvalue()

            if st.button("🔍 Analyze & Recommend"):
                # FIX: Proper progress feedback
                progress_bar = st.progress(0, text="🧠 Initializing CV analysis...")
                import time
                time.sleep(0.3)
                progress_bar.progress(30, text="📡 Sending to Vision API...")
                time.sleep(0.3)

                tags_raw, err = classify_image_with_imagga(img_bytes)

                progress_bar.progress(60, text="🔍 Matching products...")
                time.sleep(0.2)

                if tags_raw:
                    matched_pids = find_products_from_tags(tags_raw, products)
                    method = "🌐 Imagga Vision API"
                    tags = tags_raw
                else:
                    tags_raw_mock, matched_pids = mock_classify_image(image)
                    method = "🎨 Color-based CV (Demo Mode)"
                    tags = tags_raw_mock

                if not matched_pids:
                    matched_pids = ["P001","P051","P121"]

                progress_bar.progress(100, text="✅ Analysis complete!")
                time.sleep(0.3)
                progress_bar.empty()

                st.session_state["cv_tags"]   = tags
                st.session_state["cv_pids"]   = matched_pids
                st.session_state["cv_method"] = method
                st.session_state["cv_done"]   = True

    with col2:
        if st.session_state.get("cv_done"):
            method  = st.session_state["cv_method"]
            tags    = st.session_state["cv_tags"]
            matched = st.session_state["cv_pids"]

            st.markdown('<div class="section-header">🏷️ Detected Labels</div>', unsafe_allow_html=True)
            st.markdown(f'<small style="color:#7f8c8d;">Method: {method}</small>', unsafe_allow_html=True)

            # FIX: Show confidence scores with progress bars
            tags_html = ""
            for item in tags[:10]:
                if isinstance(item, dict):
                    tag_name = item["tag"]
                    conf     = item["confidence"]
                    conf_int = int(conf)
                    tags_html += f"""
                    <div style="margin:4px 0;">
                        <span class="badge badge-orange">{tag_name}</span>
                        <span class="badge badge-conf">{conf:.0f}%</span>
                        <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf_int}%;"></div></div>
                    </div>"""
                else:
                    tags_html += f'<span class="badge badge-orange">{item}</span>'
            st.markdown(f'<div style="margin:0.75rem 0;">{tags_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">🛒 Matched Products</div>', unsafe_allow_html=True)
            for pid in matched:
                p = products.get(pid)
                if p:
                    st.markdown(f"""
                    <div class="detected-item">
                        <b>{p['emoji']} {p['name']}</b>
                        <span style="float:right;color:#2ECC71;font-weight:600;">₹{p['price']}</span><br>
                        <small style="color:#7f8c8d;">Category: {p['category']}</small>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"🛒 Add {p['name']}", key=f"cv_match_{pid}"):
                        add_to_cart(pid, products)
                        st.toast(f"✅ {p['name']} added!", icon="🛒")

            if matched:
                st.markdown('<div class="section-header">🤖 CF-Enhanced Suggestions</div>', unsafe_allow_html=True)

                # FIX: Category-filtered CF suggestions
                base_cat = products.get(matched[0], {}).get("category", "")
                allowed  = RELATED_CATEGORIES.get(base_cat, [base_cat])
                sim_pids, sim_scores = get_similar_products(matched[0], item_sim_df, products, 4, filter_categories=allowed)

                sc = st.columns(2)
                for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                    if pid not in products:
                        continue
                    p = products[pid]
                    with sc[i % 2]:
                        st.markdown(f"""
                        <div class="product-card">
                            <span class="product-emoji">{p['emoji']}</span>
                            <div class="product-name">{p['name']}</div>
                            <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                            <div class="product-score">🔗 {score:.3f}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cv_cf_{pid}_{i}"):
                            add_to_cart(pid, products)
                            st.toast(f"✅ {p['name']} added!", icon="🛒")

    st.markdown("---")
    st.markdown("### 🔧 Want Real CV Power?")
    st.markdown("""
    Add your **Imagga API** credentials in Streamlit secrets:
    ```toml
    IMAGGA_API_KEY = "your_key_here"
    IMAGGA_API_SECRET = "your_secret_here"
    ```
    Get a free key at [imagga.com](https://imagga.com) — 1000 free requests/month.
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
            "Product": [f"{products[p]['emoji']} {products[p]['name']}" for p in popularity.index if p in products],
            "Buyers":  [popularity[p] for p in popularity.index if p in products]
        })
        st.bar_chart(pop_df.set_index("Product"))

    with col2:
        st.markdown('<div class="section-header">📂 Category Distribution</div>', unsafe_allow_html=True)
        cat_counts = {}
        for p in products.values():
            cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
        cat_df = pd.DataFrame({"Category": list(cat_counts.keys()), "Count": list(cat_counts.values())})
        st.bar_chart(cat_df.set_index("Category"))

    st.markdown('<div class="section-header">📈 User Purchase Heatmap (Sample — first 15 users, 10 products)</div>', unsafe_allow_html=True)
    sample = ratings_df.iloc[:15, :10].copy()
    sample.columns = [f"{products[c]['emoji']}{products[c]['name'][:8]}" for c in sample.columns if c in products]
    st.dataframe(sample.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown('<div class="section-header">🔬 SVD Explained Variance</div>', unsafe_allow_html=True)
    n_comp   = min(10, ratings_df.shape[0]-1, ratings_df.shape[1]-1)
    svd_test = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_test.fit(ratings_df.values)
    var_df = pd.DataFrame({
        "Component":              [f"C{i+1}" for i in range(n_comp)],
        "Explained Variance (%)": (svd_test.explained_variance_ratio_ * 100).round(2)
    })
    st.bar_chart(var_df.set_index("Component"))
    st.caption(f"Total variance explained: {svd_test.explained_variance_ratio_.sum()*100:.1f}%")
