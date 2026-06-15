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

st.set_page_config(
    page_title="🛒 Smart Grocery Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    .sidebar-logo { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #2ECC71 !important; text-align: center; padding: 1rem 0; }
    .conf-bar-wrap { background: #e8f8f2; border-radius: 6px; height: 6px; margin: 3px 0 6px 0; }
    .conf-bar { background: #2ECC71; height: 6px; border-radius: 6px; }
    .image-preview-box {
        border: 2px dashed #2ECC71; border-radius: 12px; padding: 0.5rem;
        background: #f8fffe; text-align: center; margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_EMOJI = {
    "Bakery": "🍪", "Snacks": "🥔", "Dairy": "🥛", "Grains": "🌾",
    "Spices": "🌶️", "Noodles": "🍜", "Drinks": "🥤", "Condiments": "🫙",
    "Personal Care": "🧴", "Health": "💊", "Home Care": "🧹",
    "Frozen": "🧊", "Beverages": "☕",
}

RELATED_CATEGORIES = {
    "Drinks":        ["Drinks"],
    "Beverages":     ["Beverages"],
    "Snacks":        ["Snacks", "Bakery"],
    "Bakery":        ["Bakery", "Snacks", "Dairy", "Condiments"],
    "Dairy":         ["Dairy", "Bakery"],
    "Grains":        ["Grains", "Spices", "Condiments", "Noodles"],
    "Spices":        ["Spices", "Grains", "Condiments", "Noodles"],
    "Noodles":       ["Noodles", "Grains", "Spices", "Condiments"],
    "Condiments":    ["Condiments", "Spices", "Grains", "Noodles"],
    "Personal Care": ["Personal Care", "Home Care"],
    "Health":        ["Health"],
    "Home Care":     ["Home Care", "Personal Care"],
    "Frozen":        ["Frozen", "Snacks"],
}

def get_emoji(category):
    return CATEGORY_EMOJI.get(category, "🛒")


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
        st.error("❌ products.csv not found!")
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
    if product_id not in item_sim_df.columns:
        return [], []
    sims = item_sim_df[product_id].drop(product_id).sort_values(ascending=False)
    if filter_categories:
        allowed_pids = [pid for pid in sims.index if products.get(pid, {}).get("category") in filter_categories]
        sims = sims[allowed_pids]
    top = sims.head(n)
    return list(top.index), list(top.values)


def init_cart():
    if "cart" not in st.session_state:
        st.session_state["cart"] = {}

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


GROCERY_KEYWORDS = {
    # ── Dairy ──
    "milk":         ["P021","P030","P136","P027"],
    "butter":       ["P021","P025","P135"],
    "ghee":         ["P025"],
    "curd":         ["P023","P029","P030"],
    "yogurt":       ["P023","P029","P030"],
    "dahi":         ["P023","P029","P030"],
    "dairy":        ["P021","P022","P023","P025","P028","P030"],
    "cream":        ["P028","P027"],
    "cheese":       ["P022"],
    "paneer":       ["P024"],
    "lassi":        ["P136"],
    "shrikhand":    ["P026"],
    "condensed":    ["P027"],
    "amul":         ["P021","P022","P025","P026","P028","P030","P136"],
    "pasteurised":  ["P021","P030"],
    "spread":       ["P021","P079"],
    "salted":       ["P021","P083"],
    "nestle":       ["P023","P029"],
    "mother dairy": ["P024","P030"],

    # ── Bakery ──
    "biscuit":      ["P001","P002","P003","P004","P005","P006","P007","P008","P009","P010"],
    "parle":        ["P001","P010"],
    "britannia":    ["P003","P080","P132","P133","P134"],
    "glucose":      ["P001","P010"],
    "marie":        ["P003"],
    "bourbon":      ["P004"],
    "chocolate biscuit": ["P004","P006","P133"],
    "cookie":       ["P006","P133","P134"],
    "digestive":    ["P007"],
    "cream biscuit":["P004","P006","P134"],
    "bread":        ["P001","P003","P132"],
    "bakery":       ["P001","P002","P003","P006","P007","P132","P133"],
    "cracker":      ["P008","P009"],
    "monaco":       ["P009"],
    "tiger":        ["P010"],

    # ── Snacks ──
    "chips":        ["P011","P012","P013","P014","P018","P143","P144"],
    "lays":         ["P011","P018","P143"],
    "kurkure":      ["P012"],
    "bingo":        ["P013"],
    "pringles":     ["P014"],
    "namkeen":      ["P015","P016","P017","P020","P141","P142"],
    "haldiram":     ["P015","P016","P017","P112","P141","P142"],
    "bhujia":       ["P015"],
    "mixture":      ["P016"],
    "moong dal":    ["P017"],
    "snack":        ["P011","P012","P013","P015","P016","P019","P020"],
    "nacho":        ["P144"],
    "doritos":      ["P144"],
    "sev":          ["P141"],
    "boondi":       ["P142"],
    "multigrain":   ["P007","P019","P037"],
    "cashew":       ["P002"],

    # ── Noodles ──
    "noodle":       ["P051","P052","P053","P054","P057","P058","P059","P131"],
    "noodles":      ["P051","P052","P053","P054","P057","P058","P059","P131"],
    "maggi":        ["P051","P057","P131"],
    "yippee":       ["P052"],
    "pasta":        ["P055","P056","P060"],
    "ramen":        ["P051","P052","P054","P131"],
    "instant noodle":["P051","P052","P053","P054","P131"],
    "instant":      ["P051","P052","P053","P131"],
    "vermicelli":   ["P052","P057"],
    "soupy":        ["P053"],
    "knorr":        ["P053"],
    "wai wai":      ["P054"],
    "patanjali noodle": ["P059"],

    # ── Grains ──
    "rice":         ["P031","P032"],
    "basmati":      ["P031"],
    "india gate":   ["P031"],
    "daawat":       ["P032"],
    "dal":          ["P033","P034","P017","P112"],
    "toor dal":     ["P033"],
    "chana dal":    ["P034"],
    "atta":         ["P035","P145"],
    "aashirvaad":   ["P035"],
    "fortune atta": ["P145"],
    "maida":        ["P036"],
    "flour":        ["P035","P036","P145"],
    "wheat":        ["P035","P145"],
    "oats":         ["P037","P038","P094","P146"],
    "oat":          ["P037","P038","P094","P146"],
    "quaker":       ["P037"],
    "saffola oats": ["P038"],
    "poha":         ["P039"],
    "suji":         ["P040"],
    "rawa":         ["P040"],
    "grain":        ["P031","P032","P033","P034"],
    "grains":       ["P031","P032","P033","P034"],
    "cereal":       ["P031","P037","P038","P094"],
    "pulses":       ["P033","P034"],
    "lentil":       ["P033","P034"],

    # ── Spices ──
    "masala":       ["P041","P042","P045","P047","P048","P147","P148"],
    "mdh":          ["P041","P045","P047","P147"],
    "everest":      ["P042","P044","P046","P048","P148"],
    "garam masala": ["P041"],
    "chilli":       ["P044"],
    "turmeric":     ["P043"],
    "haldi":        ["P043"],
    "coriander":    ["P046"],
    "dhania":       ["P046"],
    "rajma masala": ["P045"],
    "chana masala": ["P047"],
    "pav bhaji":    ["P048"],
    "biryani masala":["P147"],
    "sabji masala": ["P148"],
    "spice":        ["P041","P042","P043","P044","P147","P148"],
    "spices":       ["P041","P042","P043","P044","P147","P148"],
    "oil":          ["P049","P050"],
    "saffola oil":  ["P049"],
    "sunflower oil":["P050"],
    "fortune oil":  ["P050"],
    "cooking oil":  ["P049","P050"],

    # ── Condiments ──
    "jam":          ["P071","P080"],
    "kissan":       ["P071"],
    "sauce":        ["P072","P073","P075"],
    "ketchup":      ["P073"],
    "heinz":        ["P073"],
    "chutney":      ["P074"],
    "schezwan":     ["P074"],
    "mayo":         ["P076"],
    "mayonnaise":   ["P076"],
    "honey":        ["P077","P096"],
    "druk honey":   ["P077"],
    "nutella":      ["P078"],
    "peanut butter":["P079"],
    "pickle":       ["P072","P074"],

    # ── Drinks ──
    "juice":        ["P061","P062","P063","P066","P067","P068","P137","P138"],
    "tropicana":    ["P061","P138"],
    "real juice":   ["P062","P137","P138"],
    "frooti":       ["P063"],
    "mango drink":  ["P062","P063","P066"],
    "mango":        ["P062","P063","P066","P067"],
    "maaza":        ["P066"],
    "paper boat":   ["P067"],
    "orange juice": ["P061"],
    "pomegranate":  ["P137"],
    "guava":        ["P138"],
    "apple juice":  ["P139"],
    "energy drink": ["P064","P070"],
    "sting":        ["P064"],
    "red bull":     ["P070"],
    "soda":         ["P065","P139","P140"],
    "limca":        ["P065"],
    "7up":          ["P140"],
    "appy":         ["P139"],
    "water":        ["P069"],
    "bisleri":      ["P069"],
    "fruit":        ["P061","P062","P063","P066","P137","P138"],

    # ── Beverages ──
    "tea":          ["P121","P122","P125","P126","P130"],
    "tata tea":     ["P121"],
    "red label":    ["P122"],
    "green tea":    ["P125"],
    "lipton":       ["P125"],
    "masala chai":  ["P126"],
    "tetley":       ["P126"],
    "taj mahal tea":["P130"],
    "coffee":       ["P123","P124","P127"],
    "nescafe":      ["P123"],
    "bru":          ["P124"],
    "davidoff":     ["P127"],
    "bournvita":    ["P128"],
    "milo":         ["P129"],
    "health drink": ["P091","P092","P093","P128","P129"],
    "horlicks":     ["P091"],
    "complan":      ["P092"],
    "beverage":     ["P121","P122","P123","P124","P125"],

    # ── Personal Care ──
    "toothpaste":   ["P081"],
    "colgate":      ["P081"],
    "toothbrush":   ["P082"],
    "oral-b":       ["P082"],
    "soap":         ["P083","P084","P149"],
    "dove":         ["P083"],
    "dettol":       ["P084"],
    "pears":        ["P149"],
    "shampoo":      ["P085","P086"],
    "head shoulders":["P085"],
    "pantene":      ["P086"],
    "lotion":       ["P087"],
    "nivea":        ["P087"],
    "coconut oil":  ["P088"],
    "parachute":    ["P088"],
    "razor":        ["P089"],
    "gillette":     ["P089"],
    "facewash":     ["P085","P150"],
    "garnier":      ["P150"],
    "skincare":     ["P087","P150"],
    "personal":     ["P081","P083","P084","P085","P087"],

    # ── Home Care ──
    "detergent":    ["P101","P102"],
    "surf excel":   ["P101"],
    "ariel":        ["P102"],
    "dishwash":     ["P103"],
    "vim":          ["P103"],
    "toilet cleaner":["P104"],
    "harpic":       ["P104"],
    "glass cleaner":["P105"],
    "colin":        ["P105"],
    "floor cleaner":["P106"],
    "lizol":        ["P106"],
    "freshener":    ["P107"],
    "odonil":       ["P107"],
    "scrub":        ["P108"],
    "mosquito":     ["P109","P110"],
    "mortein":      ["P109"],
    "good knight":  ["P110"],
    "cleaner":      ["P101","P103","P105","P106"],
    "home":         ["P101","P103","P106","P108"],

    # ── Health ──
    "supplement":   ["P091","P092","P093","P098"],
    "muesli":       ["P094"],
    "chyawanprash": ["P095","P100"],
    "patanjali":    ["P095","P059","P100"],
    "dabur":        ["P096"],
    "vitamin":      ["P097"],
    "revital":      ["P097"],
    "pediasure":    ["P098"],
    "glucon":       ["P099"],
    "immunity":     ["P095","P096","P100"],
    "ayurvedic":    ["P095","P100"],
    "protein":      ["P033","P091","P093"],

    # ── Frozen ──
    "frozen":       ["P111","P112","P113","P114","P115","P116"],
    "fries":        ["P111","P118"],
    "mccain":       ["P111"],
    "dal makhani":  ["P112"],
    "paneer butter":["P113"],
    "mtr":          ["P039","P113"],
    "gulab jamun":  ["P114"],
    "gits":         ["P114"],
    "ice cream":    ["P115","P116","P119"],
    "kulfi":        ["P119"],
    "vadilal":      ["P119"],
    "amul ice":     ["P115"],
    "cornetto":     ["P116"],
    "cutlet":       ["P120"],

    # ── Visual / Generic fallbacks ──
    "packet":       ["P051","P052","P131","P011","P012","P015"],
    "yellow packet":["P051","P052","P131"],
    "envelope":     ["P051","P131"],
    "bottle":       ["P049","P050","P061","P069"],
    "can":          ["P061","P063","P064"],
    "box":          ["P001","P031","P091","P092"],
    "container":    ["P023","P025","P028","P049"],
    "yellow":       ["P051","P052","P062","P063","P131"],
    "red":          ["P044","P064","P073","P122"],
    "green":        ["P033","P046","P049","P125"],
    "white":        ["P021","P024","P030","P035"],
    "orange":       ["P061","P063","P044"],
    "brown":        ["P004","P006","P025","P133"],
    "food":         ["P061","P062","P063","P066"],
    "vegetable":    ["P033","P039","P041","P049"],
    "ripe":         ["P062","P063","P066","P137"],
    "fresh":        ["P021","P024","P030","P062"],
    "organic":      ["P033","P035","P037","P095"],
    "natural":      ["P077","P096","P095"],
    "drink":        ["P061","P062","P063","P121","P123"],
    "sweet":        ["P006","P026","P077","P078","P114"],
    "chocolate":    ["P004","P006","P078","P092","P128","P133"],
    "spicy":        ["P012","P013","P015","P041","P044"],
    "healthy":      ["P007","P029","P037","P091","P094","P096"],
    "kids":         ["P001","P005","P010","P051","P063","P091"],
    "premium":      ["P002","P014","P025","P029","P093","P127"],
    "cooking":      ["P025","P033","P035","P041","P049","P050"],

    # ── Extra aliases to catch common Gemini/HF outputs ──
    "lentils":      ["P033","P034"],
    "legume":       ["P033","P034"],
    "legumes":      ["P033","P034"],
    "pulse":        ["P033","P034"],
    "wheat flour":  ["P035","P145"],
    "instant noodles": ["P051","P052","P053","P054","P131"],
    "ramen noodles":["P051","P052","P054","P131"],
    "cup noodles":  ["P051","P052","P131"],
    "packaged food":["P051","P052","P011","P012","P001"],
    "packaged":     ["P051","P052","P011","P012","P001"],
    "processed food":["P051","P011","P012"],
    "snack food":   ["P011","P012","P013","P015"],
    "breakfast":    ["P037","P038","P039","P094","P121","P122"],
    "breakfast cereal":["P037","P038","P094"],
    "cooking ingredient":["P033","P035","P041","P049","P050"],
    "seasoning":    ["P041","P042","P043","P044","P046"],
    "condiment":    ["P071","P072","P073","P074","P076","P077","P078","P079"],
    "spread":       ["P078","P079","P021"],
    "grocery":      ["P001","P011","P021","P031","P051","P061"],
    "groceries":    ["P001","P011","P021","P031","P051","P061"],
    "food item":    ["P001","P011","P021","P031","P051","P061"],
    "indian food":  ["P033","P035","P041","P051","P121"],
    "indian":       ["P033","P035","P041","P051","P121"],
    "staple":       ["P031","P033","P035","P037","P121"],
    "staples":      ["P031","P033","P035","P037","P121"],
    "ready meal":   ["P112","P113","P114"],
    "ready to eat": ["P051","P052","P112","P113"],
    "ready-to-eat": ["P051","P052","P112","P113"],
    "soft drink":   ["P065","P139","P140"],
    "carbonated":   ["P065","P139","P140"],
    "aerated":      ["P065","P139","P140"],
    "cold drink":   ["P065","P064","P070","P139","P140"],
    "fruit juice":  ["P061","P062","P063","P066","P137","P138"],
    "mango juice":  ["P062","P063","P066"],
    "dairy product":["P021","P022","P023","P024","P025","P030"],
    "dairy products":["P021","P022","P023","P024","P025","P030"],
    "flour":        ["P035","P036","P145"],
    "powder":       ["P043","P044","P046","P081","P099"],
    "sachet":       ["P051","P052","P053","P054"],
    "pouch":        ["P051","P052","P053","P054"],
    "bag":          ["P031","P032","P035","P145"],
    "sack":         ["P031","P032","P035"],
    "jar":          ["P071","P077","P078","P079","P095"],
    "tube":         ["P081"],
}

LOW_PRIORITY_TAGS = {"food", "bottle", "yellow", "ripe", "grain", "cereal", "beverage", "drink",
                     "juice", "sweet", "grocery", "groceries", "food item", "indian", "staple",
                     "packaged", "packaged food", "processed food"}


def normalize_tag(tag: str) -> str:
    """Lowercase, strip whitespace, collapse multiple spaces."""
    return " ".join(tag.lower().strip().split())


def find_products_from_tags(tag_dicts, products):
    """
    Robust tag → product matching.
    Handles: exact match, substring match, word-level overlap.
    """
    matched_high = set()
    matched_low  = set()

    for item in tag_dicts:
        raw_tag = item["tag"] if isinstance(item, dict) else str(item)
        tag = normalize_tag(raw_tag)
        tag_words = [w for w in tag.split() if len(w) >= 3]

        for keyword, pids in GROCERY_KEYWORDS.items():
            kw = normalize_tag(keyword)
            kw_words = [w for w in kw.split() if len(w) >= 3]

            # 1. Exact match
            exact = (kw == tag)
            # 2. Substring: keyword inside tag OR tag inside keyword
            substr = (kw in tag) or (tag in kw)
            # 3. Any word of tag appears in keyword or vice-versa
            word_match = any(
                (tw in kw) or (tw in kw_words) or any(kw_w in tw for kw_w in kw_words)
                for tw in tag_words
            )

            if exact or substr or word_match:
                for pid in pids:
                    if pid in products:
                        if kw in LOW_PRIORITY_TAGS:
                            matched_low.add(pid)
                        else:
                            matched_high.add(pid)

    combined = list(matched_high)
    if len(combined) < 4:
        for p in matched_low:
            if p not in matched_high:
                combined.append(p)

    # Last-resort fallback so we never show "No products found"
    if not combined:
        fallback = ["P051","P011","P001","P021","P033","P061"]
        combined = [p for p in fallback if p in products]

    return combined[:6]


def classify_image_with_hf(image_bytes):
    import json

    # ── 1. GEMINI (primary) ──
    try:
        gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if gemini_key:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            GEMINI_MODELS = [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ]
            prompt_text = (
                "You are a smart grocery store assistant in India. "
                "Look at this image carefully and identify EXACTLY what grocery/food/household product is shown.\n\n"
                "Return ONLY a JSON array. No markdown, no extra text. Example:\n"
                '[{"tag": "maggi", "confidence": 95.0}, {"tag": "noodles", "confidence": 92.0}, '
                '{"tag": "instant noodles", "confidence": 88.0}]\n\n'
                "Rules:\n"
                "- Give 4-6 tags, most specific first\n"
                "- Use simple English grocery words like: "
                "milk, butter, ghee, curd, cheese, paneer, lassi, "
                "biscuit, bread, cookie, "
                "chips, namkeen, bhujia, sev, "
                "noodles, maggi, pasta, ramen, "
                "rice, dal, atta, flour, oats, poha, "
                "masala, turmeric, chilli, oil, "
                "juice, tea, coffee, soda, water, "
                "soap, shampoo, toothpaste, detergent, "
                "ketchup, jam, honey, peanut butter, "
                "chocolate, bournvita, horlicks, "
                "ice cream, frozen, fries\n"
                "- confidence between 0-100\n"
                "- Return ONLY the JSON array"
            )
            for gmodel in GEMINI_MODELS:
                API_URL = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{gmodel}:generateContent?key={gemini_key}"
                )
                payload = {
                    "contents": [{
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                            {"text": prompt_text}
                        ]
                    }]
                }
                try:
                    response = requests.post(API_URL, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        text = text.replace("```json", "").replace("```", "").strip()
                        # Strip any leading/trailing non-JSON characters
                        start = text.find("[")
                        end   = text.rfind("]") + 1
                        if start != -1 and end > start:
                            text = text[start:end]
                        result = json.loads(text)
                        if result and isinstance(result, list):
                            # Normalize all tags
                            for r in result:
                                if isinstance(r, dict) and "tag" in r:
                                    r["tag"] = normalize_tag(str(r["tag"]))
                            return result, None
                    elif response.status_code == 429:
                        continue  # Try next model
                    else:
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # ── 2. HUGGING FACE (fallback) ──
    try:
        hf_token = str(st.secrets.get("HF_API_TOKEN", "")).strip()
        if hf_token:
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "image/jpeg",
            }
            MODELS = [
                "google/vit-base-patch16-224",
                "microsoft/resnet-50",
            ]
            LABEL_TO_GROCERY = {
                "banana": "banana", "fig": "fruit", "pineapple": "fruit",
                "strawberry": "fruit", "orange": "orange", "lemon": "fruit",
                "apple": "apple", "mango": "mango", "grape": "fruit",
                "watermelon": "fruit", "pomegranate": "fruit", "papaya": "fruit",
                "jackfruit": "fruit", "guava": "fruit",
                "milk can": "milk", "milk": "milk", "butter": "butter",
                "cheese": "cheese", "curd": "curd", "yogurt": "yogurt",
                "cream": "dairy", "paneer": "paneer", "ghee": "ghee",
                "rice": "rice", "bread": "bread", "bagel": "bread",
                "loaf": "bread", "corn": "cereal", "wheat": "wheat",
                "oat": "oats", "grain": "grain", "flour": "flour",
                "chip": "chips", "popcorn": "snack", "cracker": "biscuit",
                "wafer": "biscuit", "pretzel": "biscuit", "nacho": "nacho",
                "cookie": "biscuit", "biscuit": "biscuit",
                "coffee": "coffee", "espresso": "coffee", "latte": "coffee",
                "tea": "tea", "juice": "juice", "smoothie": "juice",
                "bottle": "bottle", "can": "drink", "cup": "tea",
                "sauce": "ketchup", "ketchup": "ketchup", "honey": "honey",
                "jam": "jam", "oil": "oil", "pickle": "pickle",
                "noodle": "noodles", "pasta": "pasta", "spaghetti": "noodles",
                "ramen": "ramen", "instant": "maggi", "maggi": "maggi",
                "vermicelli": "noodles", "macaroni": "pasta",
                "packet": "packaged food", "envelope": "packaged food",
                "box": "box", "container": "container", "wrapper": "packaged food",
                "soap": "soap", "shampoo": "shampoo", "lotion": "lotion",
                "toothpaste": "toothpaste", "detergent": "detergent",
                "yellow": "yellow", "red": "red", "green": "green",
                "white": "white", "golden": "yellow",
                "dal": "dal", "lentil": "dal", "lentils": "dal",
                "masala": "masala", "spice": "spices", "turmeric": "turmeric",
                "atta": "atta", "roti": "atta",
            }
            for model in MODELS:
                API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
                try:
                    response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
                    if response.status_code == 200:
                        results = response.json()
                        if isinstance(results, list) and results:
                            tags = []
                            seen = set()
                            for item in results[:8]:
                                label = item.get("label", "").lower()
                                conf  = round(item.get("score", 0.0) * 100, 1)
                                if conf < 5:
                                    continue
                                grocery_tag = None
                                for key, val in LABEL_TO_GROCERY.items():
                                    if key in label:
                                        grocery_tag = val
                                        break
                                if not grocery_tag:
                                    grocery_tag = label.split(",")[0].split(" ")[0].strip()
                                grocery_tag = normalize_tag(grocery_tag)
                                if grocery_tag and grocery_tag not in seen:
                                    seen.add(grocery_tag)
                                    tags.append({"tag": grocery_tag, "confidence": conf})
                            if tags:
                                return tags, None
                except Exception:
                    continue
    except Exception:
        pass

    return None, "Both Gemini and HF Vision failed"


def fallback_color_analysis(image: Image.Image):
    img_small = image.resize((100, 100)).convert("RGB")
    pixels = np.array(img_small).reshape(-1, 3).astype(float)
    non_white_mask = ~((pixels[:, 0] > 220) & (pixels[:, 1] > 220) & (pixels[:, 2] > 220))
    fg_pixels = pixels[non_white_mask]
    if len(fg_pixels) < 50:
        fg_pixels = pixels
    avg = fg_pixels.mean(axis=0)
    r, g, b = avg
    brightness = (r + g + b) / 3
    if r > 160 and g > 130 and b < 110 and r > b * 1.7 and g > b * 1.4:
        return [{"tag": "banana", "confidence": 74.0}, {"tag": "fruit", "confidence": 70.0}]
    elif r > 190 and g > 90 and g < 170 and b < 90 and r > g * 1.2:
        return [{"tag": "mango", "confidence": 70.0}, {"tag": "fruit", "confidence": 65.0}]
    elif g > r and g > b and g > 100 and g > r * 1.1:
        return [{"tag": "vegetable", "confidence": 70.0}, {"tag": "food", "confidence": 65.0}]
    elif r > g * 1.4 and r > b * 1.4 and r > 140:
        return [{"tag": "masala", "confidence": 67.0}, {"tag": "spices", "confidence": 62.0}]
    elif b > r * 1.1 and b > g * 1.1:
        return [{"tag": "milk", "confidence": 66.0}, {"tag": "dairy", "confidence": 63.0}]
    elif brightness > 215 and r > 205 and g > 205 and b > 205:
        return [{"tag": "flour", "confidence": 65.0}, {"tag": "atta", "confidence": 62.0}]
    elif brightness < 80:
        return [{"tag": "coffee", "confidence": 68.0}, {"tag": "tea", "confidence": 65.0}]
    elif r > 130 and g > 90 and b < 90 and r > g and r > b * 1.5:
        return [{"tag": "bread", "confidence": 66.0}, {"tag": "biscuit", "confidence": 63.0}]
    else:
        return [{"tag": "snack", "confidence": 63.0}, {"tag": "packaged food", "confidence": 60.0}]


# ── SIDEBAR ──
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
    st.markdown('<p style="font-size:0.75rem;opacity:0.5;text-align:center;">Built with ❤️ using Streamlit<br>ML + CV Domain Project</p>', unsafe_allow_html=True)


# ── LOAD DATA ──
products, ratings_df = load_data()
predicted_df, item_sim_df = train_model(ratings_df)
init_cart()

users       = ratings_df.index.tolist()
product_ids = list(products.keys())
n_users     = len(users)
n_products  = len(products)


# ── PAGE: HOME ──
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">🛒 Smart Grocery Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Collaborative Filtering + Computer Vision — ML/CV Domain Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-number">{n_users}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-number">{n_products}</div><div class="stat-label">Products</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-number">SVD</div><div class="stat-label">CF Model</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-number">CV</div><div class="stat-label">Vision Module</div></div>', unsafe_allow_html=True)

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


# ── PAGE: CF RECOMMENDATIONS ──
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
                    if pid not in products: continue
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
        sel_product = st.selectbox("Select a product", product_ids, format_func=lambda x: f"{products[x]['emoji']} {products[x]['name']}")
        if st.button("🔍 Find Similar Products"):
            base_cat = products[sel_product]["category"]
            allowed  = RELATED_CATEGORIES.get(base_cat, [base_cat])
            sim_pids, sim_scores = get_similar_products(sel_product, item_sim_df, products, n_recs, filter_categories=allowed)
            st.markdown(f'<div class="section-header">Products similar to {products[sel_product]["name"]}</div>', unsafe_allow_html=True)
            if not sim_pids:
                st.info("No similar products found.")
            scols = st.columns(3)
            for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                if pid not in products: continue
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


# ── PAGE: IMAGE SCANNER ──
elif page == "📸 Image Scanner":
    st.markdown('<h1 class="main-title">📸 Product Image Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grocery photo → CV identifies it → Recommends similar products</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("📌 Upload any grocery/food product image. The CV module analyzes it and maps it to products in our catalog, then uses CF to suggest related items.")

    up_col, prev_col = st.columns([1, 1])
    with up_col:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a grocery product image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image     = Image.open(uploaded_file).convert("RGB")
        img_bytes = uploaded_file.getvalue()

        with prev_col:
            st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        btn1, btn2 = st.columns([1, 1])
        with btn1:
            analyze = st.button("🔍 Analyze & Recommend", use_container_width=True)
        with btn2:
            if st.button("🔄 Change Image", use_container_width=True):
                st.session_state["cv_done"] = False
                st.rerun()

        if analyze:
            import time
            pb = st.progress(0, text="🧠 Initializing...")
            time.sleep(0.3)
            pb.progress(25, text="🤖 Vision API analyzing image...")
            tags_raw, err = classify_image_with_hf(img_bytes)
            pb.progress(70, text="🔍 Matching products in catalog...")
            time.sleep(0.2)

            if tags_raw:
                matched_pids = find_products_from_tags(tags_raw, products)
                pb.progress(100, text="✅ Done!")
                time.sleep(0.3); pb.empty()
                st.session_state["cv_tags"]   = tags_raw
                st.session_state["cv_pids"]   = matched_pids
                st.session_state["cv_method"] = "✨ Gemini / HF Vision"
                st.session_state["cv_done"]   = True
            else:
                pb.progress(85, text="🎨 Using color-based fallback...")
                time.sleep(0.3)
                fallback_tags = fallback_color_analysis(image)
                matched_pids  = find_products_from_tags(fallback_tags, products)
                pb.progress(100, text="✅ Done!")
                time.sleep(0.3); pb.empty()
                st.session_state["cv_tags"]   = fallback_tags
                st.session_state["cv_pids"]   = matched_pids
                st.session_state["cv_method"] = "🎨 Color-Based Fallback"
                st.session_state["cv_done"]   = True
                if err:
                    st.warning(f"⚠️ Vision API error: {err}. Color fallback used.")

    if st.session_state.get("cv_done"):
        method  = st.session_state["cv_method"]
        tags    = st.session_state["cv_tags"]
        matched = st.session_state["cv_pids"]

        st.markdown("---")
        st.markdown('<div class="section-header">🏷️ Detected Labels</div>', unsafe_allow_html=True)
        st.markdown(f'<small style="color:#7f8c8d;">Method: {method}</small>', unsafe_allow_html=True)

        tags_html = ""
        for item in tags[:10]:
            if isinstance(item, dict):
                tag_name = item.get("tag", ""); conf = item.get("confidence", 0); conf_int = int(conf)
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
        if not matched:
            st.info("No matching products found. Try a different image.")
        else:
            m_cols = st.columns(3)
            for i, pid in enumerate(matched):
                p = products.get(pid)
                if not p: continue
                with m_cols[i % 3]:
                    st.markdown(f"""
                    <div class="product-card">
                        <span class="product-emoji">{p['emoji']}</span>
                        <div class="product-name">{p['name']}</div>
                        <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                        <small style="color:#7f8c8d;">{p['category']}</small>
                    </div>""", unsafe_allow_html=True)
                    if st.button("🛒 Add", key=f"cv_match_{pid}"):
                        add_to_cart(pid, products)
                        st.toast(f"✅ {p['name']} added!", icon="🛒")

        if matched:
            st.markdown('<div class="section-header">🤖 CF-Enhanced Suggestions</div>', unsafe_allow_html=True)
            base_cat = products.get(matched[0], {}).get("category", "")
            def is_relevant(pid):
                p = products.get(pid, {})
                name = p.get("name", "").lower()
                blocked = ["energy", "water", "soda", "cola", "aerated", "bisleri", "sting",
                           "paper boat", "rooh afza", "limca", "soft drink"]
                return not any(b in name for b in blocked)

            sim_pids_all, sim_scores_all = get_similar_products(
                matched[0], item_sim_df, products, n_recs * 2, filter_categories=[base_cat]
            )
            sim_pids   = [p for p in sim_pids_all if is_relevant(p)][:n_recs]
            sim_scores = [sim_scores_all[sim_pids_all.index(p)] for p in sim_pids]

            if not sim_pids:
                st.info("No CF suggestions found.")
            else:
                s_cols = st.columns(4)
                for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                    if pid not in products: continue
                    p = products[pid]
                    with s_cols[i % 4]:
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


# ── PAGE: ANALYTICS ──
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

    st.markdown('<div class="section-header">📈 User Purchase Heatmap (Sample)</div>', unsafe_allow_html=True)
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
