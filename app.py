import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import requests
import re
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

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
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    .sidebar-logo { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #2ECC71 !important; text-align: center; padding: 1rem 0; }
    .conf-bar-wrap { background: #e8f8f2; border-radius: 6px; height: 6px; margin: 3px 0 6px 0; }
    .conf-bar { background: #2ECC71; height: 6px; border-radius: 6px; }
    .image-preview-box { border: 2px dashed #2ECC71; border-radius: 12px; padding: 0.5rem; background: #f8fffe; text-align: center; margin-bottom: 0.75rem; }
    .detected-product-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #2ECC71; border-radius: 12px; padding: 1rem 1.5rem;
        margin: 1rem 0; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
CATEGORY_EMOJI = {
    "Bakery": "🍪", "Snacks": "🥔", "Dairy": "🥛", "Grains": "🌾",
    "Spices": "🌶️", "Noodles": "🍜", "Drinks": "🥤", "Condiments": "🫙",
    "Personal Care": "🧴", "Health": "💊", "Home Care": "🧹",
    "Frozen": "🧊", "Beverages": "☕",
}

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

def get_emoji(cat):
    return CATEGORY_EMOJI.get(cat, "🛒")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=0)
def load_data():
    products_df = None
    for path in ["data/products.csv", "./data/products.csv", "products.csv"]:
        if os.path.exists(path):
            products_df = pd.read_csv(path)
            break
    if products_df is None:
        st.error("❌ products.csv not found!")
        st.stop()

    products = {}
    for _, row in products_df.iterrows():
        pid   = row["product_id"]
        tags  = [t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()]
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

    user_ids = [f"U{str(i).zfill(3)}" for i in range(1, 51)]
    try:
        new_df = None
        for p in ["data/users_new.csv", "./data/users_new.csv", "users_new.csv"]:
            if os.path.exists(p):
                new_df = pd.read_csv(p)
                break
        if new_df is not None:
            user_ids += new_df["user_id"].tolist()
    except Exception:
        pass

    product_ids  = list(products.keys())
    ratings_path = next(
        (p for p in ["data/ratings.csv", "./data/ratings.csv", "ratings.csv"] if os.path.exists(p)),
        None
    )
    try:
        if not ratings_path:
            raise FileNotFoundError
        raw    = pd.read_csv(ratings_path)
        matrix = raw.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="mean")
        matrix = matrix.reindex(index=user_ids, columns=product_ids, fill_value=0).fillna(0)
    except FileNotFoundError:
        np.random.seed(42)
        data = np.random.choice(
            [0, 0, 0, 1, 2, 3, 4, 5],
            size=(len(user_ids), len(product_ids)),
            p=[0.5, 0.1, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02]
        )
        matrix = pd.DataFrame(data, index=user_ids, columns=product_ids)
    return products, matrix


# ═══════════════════════════════════════════════════════════════
# CF MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def train_model(_df):
    n_components = min(20, _df.shape[0] - 1, _df.shape[1] - 1)
    svd          = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(_df.values)
    predicted    = np.dot(user_factors, svd.components_)
    predicted_df = pd.DataFrame(predicted, index=_df.index, columns=_df.columns)
    item_sim     = cosine_similarity(svd.components_.T)
    item_sim_df  = pd.DataFrame(item_sim, index=_df.columns, columns=_df.columns)
    return predicted_df, item_sim_df


def get_user_recommendations(user_id, df, predicted_df, n=6):
    bought = df.loc[user_id][df.loc[user_id] > 0].index.tolist()
    preds  = predicted_df.loc[user_id].copy()
    preds[bought] = -999
    return preds.nlargest(n).index.tolist(), bought


def get_similar_products(product_id, item_sim_df, products, n=6, filter_categories=None):
    if product_id not in item_sim_df.columns:
        return [], []
    sims = item_sim_df[product_id].drop(product_id).sort_values(ascending=False)
    if filter_categories:
        allowed_ids   = [pid for pid in sims.index if products.get(pid, {}).get("category") in filter_categories]
        filtered_sims = sims[allowed_ids]
        if len(filtered_sims) >= n:
            top = filtered_sims.head(n)
            return list(top.index), list(top.values)
    top = sims.head(n)
    return list(top.index), list(top.values)


# ═══════════════════════════════════════════════════════════════
# CART
# ═══════════════════════════════════════════════════════════════
def init_cart():
    if "cart" not in st.session_state:
        st.session_state["cart"] = {}

def add_to_cart(pid, products):
    init_cart()
    if pid in st.session_state["cart"]:
        st.session_state["cart"][pid]["qty"] += 1
    else:
        p = products[pid]
        st.session_state["cart"][pid] = {
            "name": p["name"], "price": p["price"],
            "emoji": p["emoji"], "qty": 1
        }

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


# ═══════════════════════════════════════════════════════════════
# PRODUCT KEYWORD MAP  (pid → keywords for reverse lookup)
# ═══════════════════════════════════════════════════════════════
GROCERY_KEYWORDS = {
    # ── Dairy ──
    "milk":["P021","P030","P136","P027"],
    "butter":["P021","P025"], "amul butter":["P021"],
    "salted butter":["P021"], "unsalted butter":["P021"],
    "table butter":["P021"], "cooking butter":["P021","P025"],
    "makhan":["P021","P025"], "makkhan":["P021","P025"],
    "ghee":["P025"], "clarified butter":["P025"], "desi ghee":["P025"],
    "curd":["P023","P029","P030"], "yogurt":["P023","P029","P030"],
    "dahi":["P023","P029","P030"], "yoghurt":["P023","P029","P030"],
    "dairy":["P021","P022","P023","P025","P028","P030"],
    "cream":["P028","P027"], "fresh cream":["P028"],
    "whipping cream":["P028"], "cheese":["P022"],
    "processed cheese":["P022"], "cheese slice":["P022"],
    "paneer":["P024"], "cottage cheese":["P024"],
    "lassi":["P136"], "buttermilk":["P136"],
    "shrikhand":["P026"], "condensed milk":["P027"],
    "amul":["P021","P022","P025","P026","P028","P030","P136"],
    "nestle":["P023","P029"], "mother dairy":["P024","P030"],
    "dairy product":["P021","P022","P023","P024","P025","P030"],
    "dairy products":["P021","P022","P023","P024","P025","P030"],
    # ── Bakery ──
    "biscuit":["P001","P002","P003","P004","P005","P006","P007","P008","P009","P010"],
    "parle":["P001","P010"], "britannia":["P003","P080","P132","P133","P134"],
    "marie":["P003"], "marie biscuit":["P003"],
    "bourbon":["P004"], "cookie":["P006","P133","P134"],
    "digestive":["P007"], "digestive biscuit":["P007"],
    "bread":["P001","P003","P132"], "cracker":["P008","P009"],
    "monaco":["P009"], "tiger biscuit":["P010"],
    "multigrain":["P007","P019","P037"],
    "cashew biscuit":["P002"], "bakery":["P001","P002","P003","P006","P007"],
    # ── Snacks ──
    "chips":["P011","P012","P013","P014","P018","P143","P144"],
    "lays":["P011","P018","P143"], "kurkure":["P012"],
    "bingo":["P013"], "pringles":["P014"],
    "namkeen":["P015","P016","P017","P020","P141","P142"],
    "haldiram":["P015","P016","P017","P112","P141","P142"],
    "bhujia":["P015"], "mixture":["P016"], "moong dal":["P017"],
    "snack":["P011","P012","P013","P015","P016","P019","P020"],
    "nacho":["P144"], "doritos":["P144"],
    "sev":["P141"], "boondi":["P142"], "popcorn":["P019","P020"],
    "potato chips":["P011","P012","P013","P014"],
    "corn chips":["P013","P144"],
    # ── Noodles ──
    "noodle":["P051","P052","P053","P054","P057","P058","P059","P131"],
    "noodles":["P051","P052","P053","P054","P057","P058","P059","P131"],
    "maggi":["P051","P057","P131"], "yippee":["P052"],
    "pasta":["P055","P056","P060"],
    "ramen":["P051","P052","P054","P131"],
    "instant noodle":["P051","P052","P053","P054","P131"],
    "instant noodles":["P051","P052","P053","P054","P131"],
    "vermicelli":["P052","P057"], "knorr":["P053"],
    "wai wai":["P054"], "patanjali noodle":["P059"],
    "spaghetti":["P055","P056"],
    "cup noodles":["P051","P052","P131"],
    "hakka noodles":["P058"],
    "chow mein":["P058"],
    # ── Grains ──
    "rice":["P031","P032"], "basmati":["P031"],
    "basmati rice":["P031"], "india gate":["P031"], "daawat":["P032"],
    "dal":["P033","P034","P017","P112"],
    "toor dal":["P033"], "chana dal":["P034"],
    "atta":["P035","P145"], "aashirvaad":["P035"],
    "fortune atta":["P145"], "maida":["P036"],
    "wheat flour":["P035","P145"],
    "oats":["P037","P038","P094","P146"],
    "oat":["P037","P038","P094","P146"],
    "quaker":["P037"], "saffola oats":["P038"],
    "poha":["P039"], "suji":["P040"], "rawa":["P040"],
    "lentil":["P033","P034"], "lentils":["P033","P034"],
    "legume":["P033","P034"], "pulses":["P033","P034"],
    "grain":["P031","P032","P033","P034"],
    "grains":["P031","P032","P033","P034"],
    # ── Spices ──
    "masala":["P041","P042","P045","P047","P048","P147","P148"],
    "mdh":["P041","P045","P047","P147"],
    "everest":["P042","P044","P046","P048","P148"],
    "garam masala":["P041"],
    "chilli powder":["P044"], "chili powder":["P044"],
    "red chilli":["P044"], "turmeric":["P043"], "haldi":["P043"],
    "coriander powder":["P046"], "dhania":["P046"],
    "rajma masala":["P045"], "chana masala":["P047"],
    "pav bhaji masala":["P048"], "biryani masala":["P147"],
    "sabji masala":["P148"],
    "spice":["P041","P042","P043","P044","P147","P148"],
    "spices":["P041","P042","P043","P044","P147","P148"],
    "curry powder":["P041","P042","P148"],
    # ── Oil ──
    "cooking oil":["P049","P050"], "saffola oil":["P049"],
    "sunflower oil":["P050"], "fortune oil":["P050"],
    "oil":["P049","P050"],
    # ── Condiments ──
    "jam":["P071","P080"], "kissan":["P071"],
    "ketchup":["P073"], "tomato ketchup":["P073"], "heinz":["P073"],
    "sauce":["P072","P073","P075"],
    "chutney":["P074"], "schezwan":["P074"],
    "mayonnaise":["P076"], "mayo":["P076"],
    "honey":["P077","P096"], "dabur honey":["P077"],
    "nutella":["P078"], "peanut butter":["P079"],
    "pickle":["P072","P074"],
    "condiment":["P071","P072","P073","P074","P076","P077","P078","P079"],
    # ── Drinks ──
    "juice":["P061","P062","P063","P066","P067","P068","P137","P138"],
    "tropicana":["P061","P138"],
    "real juice":["P062","P137","P138"],
    "frooti":["P063"],
    "mango drink":["P062","P063","P066"],
    "mango juice":["P062","P063","P066"],
    "maaza":["P066"], "paper boat":["P067"],
    "orange juice":["P061"],
    "pomegranate juice":["P137"], "guava juice":["P138"],
    "apple juice":["P139"],
    "energy drink":["P064","P070"],
    "sting":["P064"], "red bull":["P070"],
    "soda":["P065","P139","P140"], "limca":["P065"],
    "7up":["P140"], "appy":["P139"],
    "water":["P069"], "bisleri":["P069"],
    "cold drink":["P065","P064","P070","P139","P140"],
    "soft drink":["P065","P139","P140"],
    "fruit juice":["P061","P062","P063","P066","P137","P138"],
    "mango":["P062","P063","P066","P067"],
    "fruit":["P061","P062","P063","P066","P137","P138"],
    # ── Beverages ──
    "tea":["P121","P122","P125","P126","P130"],
    "tata tea":["P121"], "red label":["P122"],
    "green tea":["P125"], "lipton":["P125"],
    "masala chai":["P126"], "tetley":["P126"],
    "taj mahal tea":["P130"],
    "coffee":["P123","P124","P127"],
    "nescafe":["P123"], "bru":["P124"], "davidoff":["P127"],
    "bournvita":["P128"], "milo":["P129"],
    "horlicks":["P091"], "complan":["P092"],
    "health drink":["P091","P092","P093","P128","P129"],
    # ── Personal Care ──
    "toothpaste":["P081"], "colgate":["P081"],
    "toothbrush":["P082"], "oral-b":["P082"],
    "soap":["P083","P084","P149"],
    "dove":["P083"], "dettol":["P084"], "pears soap":["P149"],
    "shampoo":["P085","P086"], "pantene":["P086"],
    "body lotion":["P087"], "nivea":["P087"],
    "coconut oil":["P088"], "parachute":["P088"],
    "razor":["P089"], "gillette":["P089"],
    "face wash":["P085","P150"], "garnier":["P150"],
    # ── Home Care ──
    "detergent":["P101","P102"],
    "surf excel":["P101"], "ariel":["P102"],
    "dishwash":["P103"], "vim":["P103"],
    "toilet cleaner":["P104"], "harpic":["P104"],
    "glass cleaner":["P105"], "colin":["P105"],
    "floor cleaner":["P106"], "lizol":["P106"],
    "freshener":["P107"], "odonil":["P107"],
    "scrub":["P108"],
    "mosquito repellent":["P109","P110"],
    "mortein":["P109"], "good knight":["P110"],
    # ── Health ──
    "muesli":["P094"], "chyawanprash":["P095","P100"],
    "patanjali":["P095","P059","P100"], "dabur":["P096"],
    "vitamin":["P097"], "revital":["P097"],
    "pediasure":["P098"], "glucon d":["P099"],
    "supplement":["P091","P092","P093","P098"],
    # ── Frozen ──
    "frozen fries":["P111","P118"], "mccain":["P111"],
    "dal makhani":["P112"],
    "paneer butter masala":["P113"],
    "mtr":["P039","P113"],
    "gulab jamun":["P114"], "gits":["P114"],
    "ice cream":["P115","P116","P119"],
    "kulfi":["P119"], "vadilal":["P119"],
    "amul ice cream":["P115"], "cornetto":["P116"],
    "cutlet":["P120"], "frozen":["P111","P112","P113","P114","P115","P116"],
    # ── Generic fallbacks ──
    "packaged food":["P051","P052","P011","P012","P001"],
    "breakfast":["P037","P038","P039","P094","P121","P122"],
    "ready to eat":["P051","P052","P112","P113"],
    "indian food":["P033","P035","P041","P051","P121"],
    "staple":["P031","P033","P035","P037","P121"],
}

LOW_PRIORITY_TAGS = {
    "food","bottle","ripe","grain","cereal","beverage","juice","sweet",
    "grocery","groceries","food item","indian","staple","packaged",
    "packaged food","processed food","dairy product","dairy products",
    "cooking ingredient","indian food","breakfast","ready to eat",
    "condiment","fruit juice","staples","drink",
}

JUNK_DESCRIPTORS = {
    "block","slab","rectangular","foil","wrapped","solid","object","thing",
    "item","material","texture","surface","background","pattern","shape",
    "band","bandage","adhesive","band aid","medical","first aid","plaster",
    "strip","ribbon","tape",
    "yellow","red","green","white","orange","brown","golden","black",
    "blue","purple","pink","light","dark","bright","shiny","glossy",
    "matte","pale","beige","ivory",
    "small","large","big","round","square","flat","thick","thin",
    "packaging","wrapper","label","sticker","logo",
    "fat","spread","saturated",
    "mix","mixed","blend","product","ingredient","ingredients",
    "donut","doughnut","donuts","doughnuts","pretzel","ring",
    "coil","spiral","loop","oval","circle","ellipse",
    "fried","grilled","boiled","baked","roasted","steamed",
    "stir fry","stir-fry","deep fried",
}

CUISINE_NOISE_WORDS = {
    "greek","mediterranean","italian","indian cuisine","chinese cuisine",
    "mexican","thai cuisine","japanese cuisine","korean","american",
    "continental","fusion","ethnic","regional","homemade","restaurant",
    "dish","meal","cuisine","platter","recipe","delicacy",
}

DAIRY_SPECIFIC = {
    "butter": {
        "tags": {"butter","amul butter","salted butter","unsalted butter",
                 "table butter","cooking butter","makhan","makkhan","margarine",
                 "butter block","dairy butter"},
        "pids": ["P021","P025"],
        "blocks": {"cream","yogurt","curd","dahi","milk","lassi",
                   "paneer","cheese","ghee","condensed milk"},
    },
    "ghee": {
        "tags": {"ghee","clarified butter","desi ghee","pure ghee"},
        "pids": ["P025"],
        "blocks": {"butter","cream","yogurt","curd","milk","paneer",
                   "cheese","condensed milk"},
    },
    "curd": {
        "tags": {"curd","dahi","yogurt","yoghurt","greek yogurt",
                 "plain yogurt","set curd","thick curd"},
        "pids": ["P023","P029","P030"],
        "blocks": {"butter","cream","milk","paneer","cheese","ghee",
                   "condensed milk"},
    },
    "milk": {
        "tags": {"milk","toned milk","full cream milk","skimmed milk",
                 "double toned milk"},
        "pids": ["P021","P030","P136"],
        "blocks": {"butter","cream","curd","yogurt","paneer","cheese",
                   "ghee","condensed milk"},
    },
    "cream": {
        "tags": {"fresh cream","whipping cream","whipped cream","sour cream",
                 "cooking cream"},
        "pids": ["P028","P027"],
        "blocks": {"butter","curd","yogurt","paneer","cheese","ghee","milk"},
    },
    "cheese": {
        "tags": {"cheese","processed cheese","cream cheese","cheddar",
                 "cheese slice","cheese slices"},
        "pids": ["P022"],
        "blocks": {"butter","curd","yogurt","cream","ghee","milk","paneer"},
    },
    "paneer": {
        "tags": {"paneer","cottage cheese"},
        "pids": ["P024"],
        "blocks": {"butter","curd","yogurt","cream","ghee","milk","cheese"},
    },
}

VISUAL_LABEL_TO_TAG = {
    "noodle":"noodles","noodles":"noodles","ramen":"noodles",
    "chow mein":"noodles","lo mein":"noodles",
    "pad thai":"noodles","spaghetti":"noodles","linguine":"noodles",
    "pho":"noodles","udon":"noodles","vermicelli":"vermicelli",
    "maggi":"maggi","pasta":"pasta","penne":"pasta",
    "macaroni":"pasta","fusilli":"pasta","instant noodles":"maggi",
    "biscuit":"biscuit","cookie":"biscuit","cracker":"biscuit",
    "wafer":"biscuit","bread":"bread","loaf":"bread",
    "toast":"bread","bun":"bread","naan":"bread",
    "roti":"bread","chapati":"bread","paratha":"bread",
    "chocolate bar":"biscuit","waffle":"biscuit",
    "potato chip":"chips","potato chips":"chips","chip":"chips",
    "crisp":"chips","crisps":"chips","tortilla chip":"nacho",
    "corn chip":"chips","popcorn":"snack",
    "bhujia":"bhujia","namkeen":"namkeen","boondi":"boondi",
    "french fries":"frozen fries","nachos":"nacho",
    "butter":"butter","margarine":"butter",
    "ghee":"ghee",
    "curd":"curd","yogurt":"yogurt","yoghurt":"yogurt",
    "cheese":"cheese","paneer":"paneer","cottage cheese":"paneer",
    "milk":"milk","fresh cream":"cream","whipping cream":"cream",
    "lassi":"lassi","buttermilk":"lassi",
    "ice cream":"ice cream","gelato":"ice cream","kulfi":"kulfi",
    "rice":"rice","basmati rice":"basmati",
    "dal":"dal","lentil":"lentil","lentils":"lentil",
    "porridge":"oats","oatmeal":"oats","oats":"oats",
    "granola":"muesli","muesli":"muesli",
    "poha":"poha","upma":"suji",
    "turmeric":"turmeric","chilli":"chilli powder",
    "chili":"chilli powder","masala":"masala",
    "curry powder":"masala","garam masala":"garam masala",
    "cooking oil":"cooking oil","sunflower oil":"sunflower oil",
    "ketchup":"ketchup","mayonnaise":"mayonnaise",
    "honey":"honey","jam":"jam","jelly":"jam",
    "peanut butter":"peanut butter","nutella":"nutella",
    "chutney":"chutney","pickle":"pickle",
    "mango juice":"mango juice","orange juice":"orange juice",
    "lemonade":"soda","soda":"soda","water":"water",
    "energy drink":"energy drink",
    "tea":"tea","chai":"tea","green tea":"green tea",
    "coffee":"coffee","espresso":"coffee",
    "bournvita":"bournvita","horlicks":"horlicks","milo":"milo",
    "soap":"soap","shampoo":"shampoo","toothpaste":"toothpaste",
    "toothbrush":"toothbrush","face wash":"face wash","razor":"razor",
    "detergent":"detergent","dishwash":"dishwash",
    "toilet cleaner":"toilet cleaner","floor cleaner":"floor cleaner",
    "flour":"atta","wheat flour":"atta","whole wheat":"atta",
    "atta":"atta","maida":"maida",
    "tube":"toothpaste","sauce":"sauce","syrup":"honey","powder":"masala",
}


def normalize_tag(tag: str) -> str:
    return " ".join(tag.lower().strip().split())


def detect_dairy_type(tag_texts):
    priority_order = ["butter","ghee","paneer","cheese","curd","cream","milk"]
    found = {}
    for tag in tag_texts:
        for dtype, info in DAIRY_SPECIFIC.items():
            for known_tag in info["tags"]:
                if known_tag == tag or known_tag in tag or tag in known_tag:
                    found[dtype] = info
                    break
    for dtype in priority_order:
        if dtype in found:
            return dtype, found[dtype]["pids"], found[dtype]["blocks"]
    return None, [], set()


def find_products_from_tags(tag_dicts, products):
    """
    Improved product matching:
    - Exact match gets full confidence score
    - Substring match gets 80%
    - Word-level match gets 60%
    - Low-priority tags only used as fallback
    - Returns up to 6 best matched products
    """
    matched_high = {}
    matched_low  = set()

    raw_tags = []
    for item in tag_dicts:
        t    = item["tag"] if isinstance(item, dict) else str(item)
        conf = float(item.get("confidence", 50)) if isinstance(item, dict) else 50.0
        raw_tags.append((normalize_tag(t), conf))

    tag_texts = [t for t, _ in raw_tags]

    # Step 1: dairy-specific forced match
    dairy_type, forced_pids, blocked_tags = detect_dairy_type(tag_texts)
    for pid in forced_pids:
        if pid in products:
            matched_high[pid] = matched_high.get(pid, 0) + 200

    # Step 2: keyword matching with improved scoring
    for (tag, conf) in raw_tags:
        if tag in JUNK_DESCRIPTORS or tag in CUISINE_NOISE_WORDS:
            continue
        if dairy_type and tag in blocked_tags:
            continue

        tag_words = [w for w in tag.split() if len(w) >= 3]

        for keyword, pids in GROCERY_KEYWORDS.items():
            kw       = normalize_tag(keyword)
            kw_words = [w for w in kw.split() if len(w) >= 3]
            exact    = (kw == tag)
            substr   = (len(kw) >= 4 and kw in tag) or (len(tag) >= 4 and tag in kw)
            word_hit = any(
                (tw in kw) or any(kw_w in tw for kw_w in kw_words)
                for tw in tag_words
            )

            if exact or substr or word_hit:
                for pid in pids:
                    if pid not in products:
                        continue
                    if kw in LOW_PRIORITY_TAGS:
                        matched_low.add(pid)
                    else:
                        if exact:
                            score = conf * 1.0
                        elif substr:
                            score = conf * 0.8
                        else:
                            score = conf * 0.6
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
    raw = raw.split(",")[0].strip()
    raw = raw.split("/")[0].strip()
    raw = raw.replace("_", " ").strip()

    if not raw or len(raw) < 3:
        return ""
    if raw in JUNK_DESCRIPTORS or raw in CUISINE_NOISE_WORDS:
        return ""
    if raw in VISUAL_LABEL_TO_TAG:
        return VISUAL_LABEL_TO_TAG[raw]
    for key, val in VISUAL_LABEL_TO_TAG.items():
        if len(key) >= 5 and key in raw:
            return val
    for key, val in VISUAL_LABEL_TO_TAG.items():
        if len(raw) >= 5 and raw in key:
            return val
    first = raw.split()[0] if raw.split() else ""
    if len(first) >= 4 and first not in JUNK_DESCRIPTORS and first not in CUISINE_NOISE_WORDS:
        if first in VISUAL_LABEL_TO_TAG:
            return VISUAL_LABEL_TO_TAG[first]
    return ""


# ═══════════════════════════════════════════════════════════════
# GEMINI PROMPT — IMPROVED
# ═══════════════════════════════════════════════════════════════
GEMINI_PROMPT = """You are an expert Indian grocery product identifier.

Look at this image carefully and identify EXACTLY what grocery/food/household product is shown.

Return ONLY a valid JSON array with 4-8 tags. No explanation, no markdown, just JSON.

FORMAT: [{"tag": "product_name", "confidence": 90}, ...]

IDENTIFICATION RULES:
1. Brand name first (if visible): Amul, Britannia, Parle, Nestle, Maggi, Haldiram, MDH, Everest, Tata, Lays, Kurkure, Tropicana, Real, Frooti, Dabur, Patanjali, Colgate, Dove, Surf Excel, etc.
2. Specific product type second
3. Category last

STRICT DAIRY RULES (very important — do not confuse these):
- Rectangular solid block in paper/foil wrapper (yellow or white) → "butter" (NEVER yogurt, cream, or cheese)
- White/cream liquid in packet or bottle → "milk"
- Semi-solid white substance in cup or tub → "curd" or "yogurt"
- Thick golden/amber liquid in jar or tin → "ghee"
- Soft white block submerged in water or wrapped in cloth → "paneer"
- Pale yellow firm slices or block → "cheese"

STRICT NOODLE RULES:
- Flat/square packet with dry noodle cake visible = "instant noodles" or "maggi"
- NEVER call noodles a "donut", "ring", "pretzel", "coil", "paella"
- Orange packet with Nestle logo = "maggi"
- Yellow/red packet with Sun logo = "yippee"

STRICT SNACK RULES:
- Metallic pouch with potato slices = "chips" or "lays"
- Puffed corn curls in yellow/orange pack = "kurkure"
- Round tin with stacked chips = "pringles"
- Mixed fried Indian snack = "namkeen"

ALLOWED TAGS (use these exact words):
DAIRY: milk, butter, ghee, curd, yogurt, paneer, cheese, cream, lassi, condensed milk, ice cream, kulfi
BAKERY: biscuit, bread, cookie, digestive biscuit, bourbon, marie biscuit, cracker, cake
SNACKS: chips, lays, kurkure, bingo, pringles, namkeen, bhujia, sev, boondi, nacho, popcorn
NOODLES: maggi, instant noodles, yippee, noodles, pasta, vermicelli, ramen, cup noodles
GRAINS: rice, basmati rice, dal, toor dal, chana dal, atta, maida, oats, poha, suji
SPICES: masala, turmeric, chilli powder, garam masala, biryani masala, coriander powder, rajma masala
OIL: cooking oil, sunflower oil, coconut oil
CONDIMENTS: ketchup, jam, honey, mayonnaise, peanut butter, nutella, chutney, pickle, sauce
DRINKS: juice, mango juice, orange juice, energy drink, soda, water, frooti, maaza, sting, red bull
BEVERAGES: tea, green tea, coffee, bournvita, horlicks, milo, masala chai
PERSONAL CARE: soap, shampoo, toothpaste, toothbrush, face wash, body lotion, razor, coconut oil
HOME CARE: detergent, dishwash, toilet cleaner, floor cleaner, glass cleaner, mosquito repellent
HEALTH: muesli, oats, chyawanprash, vitamin, supplement, glucon d

FORBIDDEN TAGS (never output these):
band, bandage, block, slab, foil, wrapper, yellow, red, white, blue, strip, ribbon, tape, shape,
fried, grilled, baked, donut, ring, coil, spiral, oval, paella, risotto, rectangular, solid, packaging

confidence is an integer 0-100. Higher = more certain.
Return ONLY the JSON array."""


def classify_image_with_gemini(image_bytes):
    """Try Gemini models in order. Returns (tags_list, error_str)."""
    debug = []
    gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    if not gemini_key:
        return None, "GEMINI_API_KEY not set", debug

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    for gmodel in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
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
                    debug.append(f"⚠️ {gmodel}: empty candidates")
                    continue
                fr = cands[0].get("finishReason", "")
                if fr in ("SAFETY", "RECITATION"):
                    debug.append(f"⚠️ {gmodel}: blocked ({fr})")
                    continue
                text = cands[0]["content"]["parts"][0]["text"].strip()
                text = text.replace("```json","").replace("```","").strip()
                s, e = text.find("["), text.rfind("]") + 1
                if s != -1 and e > s:
                    text = text[s:e]
                result = json.loads(text)
                if result and isinstance(result, list):
                    cleaned = []
                    for item in result:
                        if isinstance(item, dict) and "tag" in item:
                            tag = normalize_tag(str(item["tag"]))
                            if tag in JUNK_DESCRIPTORS or tag in CUISINE_NOISE_WORDS:
                                continue
                            mapped = VISUAL_LABEL_TO_TAG.get(tag, tag)
                            if mapped and mapped not in JUNK_DESCRIPTORS and len(mapped) >= 3:
                                item["tag"] = mapped
                                cleaned.append(item)
                    if cleaned:
                        debug.append(f"✅ Gemini ({gmodel}) — {len(cleaned)} tags")
                        return cleaned, None, debug
            elif resp.status_code == 429:
                debug.append(f"⚠️ {gmodel}: rate limit")
                continue
            elif resp.status_code == 400:
                err = resp.json().get("error", {}).get("message", "")
                debug.append(f"❌ {gmodel}: bad request — {err}")
                break
            else:
                debug.append(f"❌ {gmodel}: HTTP {resp.status_code}")
                break
        except json.JSONDecodeError as je:
            debug.append(f"⚠️ {gmodel}: JSON parse error — {je}")
            continue
        except Exception as ex:
            debug.append(f"⚠️ {gmodel}: {str(ex)[:80]}")
            continue

    return None, "Gemini failed", debug


def classify_image_with_hf(image_bytes):
    """Try HuggingFace models. Returns (tags_list, error_str, debug_list)."""
    debug = []
    hf_token = str(st.secrets.get("HF_API_TOKEN", "")).strip()
    if not hf_token:
        return None, "HF_API_TOKEN not set", debug

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
                        if conf < 4:
                            continue
                        gtag = map_visual_label(raw)
                        if not gtag or len(gtag) < 3:
                            continue
                        gtag = normalize_tag(gtag)
                        if gtag not in seen:
                            seen.add(gtag)
                            tags.append({"tag": gtag, "confidence": conf})
                    if tags:
                        debug.append(f"✅ HF ({model}) — {len(tags)} tags")
                        return tags, None, debug
                    else:
                        debug.append(f"⚠️ HF ({model}): no usable tags after mapping")
            elif resp.status_code in (503, 429):
                debug.append(f"⚠️ HF ({model}): {resp.status_code}")
                continue
            else:
                debug.append(f"❌ HF ({model}): HTTP {resp.status_code}")
                continue
        except Exception as ex:
            debug.append(f"⚠️ HF ({model}): {str(ex)[:60]}")
            continue

    return None, "HF Vision failed", debug


def classify_image(image_bytes):
    """
    Full classification pipeline:
    1. Gemini (best accuracy)
    2. HuggingFace (fallback)
    3. Color analysis (last resort)
    Returns (tags_list, method_label, debug_list)
    """
    all_debug = []

    # Try Gemini first
    tags, err, dbg = classify_image_with_gemini(image_bytes)
    all_debug.extend(dbg)
    if tags:
        return tags, "✨ Gemini Vision", all_debug

    # Try HuggingFace
    tags, err, dbg = classify_image_with_hf(image_bytes)
    all_debug.extend(dbg)
    if tags:
        return tags, "🤗 HuggingFace Vision", all_debug

    # Color-based fallback
    all_debug.append("⚠️ Both APIs failed — using color fallback")
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    tags  = fallback_color_analysis(image)
    return tags, "🎨 Color Fallback", all_debug


def fallback_color_analysis(image: Image.Image):
    img  = image.resize((100, 100)).convert("RGB")
    pix  = np.array(img).reshape(-1, 3).astype(float)
    mask = ~((pix[:,0]>220) & (pix[:,1]>220) & (pix[:,2]>220))
    fg   = pix[mask] if mask.sum() > 50 else pix
    r, g, b = fg.mean(axis=0)
    bright   = (r + g + b) / 3
    if r > 160 and g > 130 and b < 110 and r > b * 1.7:
        return [{"tag":"mango juice","confidence":65}, {"tag":"juice","confidence":60}]
    if r > 190 and 90 < g < 170 and b < 90 and r > g * 1.2:
        return [{"tag":"mango juice","confidence":65}, {"tag":"juice","confidence":60}]
    if g > r and g > b and g > 100 and g > r * 1.1:
        return [{"tag":"packaged food","confidence":55}]
    if r > g * 1.4 and r > b * 1.4 and r > 140:
        return [{"tag":"chilli powder","confidence":65}, {"tag":"masala","confidence":60}]
    if b > r * 1.1 and b > g * 1.1:
        return [{"tag":"milk","confidence":62}, {"tag":"dairy product","confidence":58}]
    if bright > 210 and abs(r-g) < 20 and abs(g-b) < 20:
        return [{"tag":"butter","confidence":65}, {"tag":"dairy product","confidence":60}]
    if bright < 80:
        return [{"tag":"coffee","confidence":62}, {"tag":"tea","confidence":58}]
    if r > 130 and g > 90 and b < 90 and r > g and r > b * 1.5:
        return [{"tag":"biscuit","confidence":60}]
    return [{"tag":"snack","confidence":55}, {"tag":"packaged food","confidence":52}]


# ═══════════════════════════════════════════════════════════════
# SMART RECOMMENDATIONS after detection
# ═══════════════════════════════════════════════════════════════
def get_smart_recommendations(matched_pids, products, item_sim_df, n=8):
    """
    For each matched product, find CF similar products.
    Aggregate and de-duplicate, sorted by average similarity score.
    """
    if not matched_pids:
        return [], []

    score_map = {}
    matched_set = set(matched_pids)

    for base_pid in matched_pids[:3]:   # use top 3 matched products as seeds
        base_cat     = products.get(base_pid, {}).get("category", "")
        allowed_cats = RELATED_CATEGORIES.get(base_cat, [base_cat])
        sim_pids, sim_scores = get_similar_products(
            base_pid, item_sim_df, products, n * 2, filter_categories=allowed_cats
        )
        for pid, score in zip(sim_pids, sim_scores):
            if pid not in matched_set:
                if pid in score_map:
                    score_map[pid] = max(score_map[pid], score)
                else:
                    score_map[pid] = score

    # Sort by score descending
    sorted_pids   = sorted(score_map.keys(), key=lambda p: score_map[p], reverse=True)
    top_pids      = sorted_pids[:n]
    top_scores    = [score_map[p] for p in top_pids]
    return top_pids, top_scores


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════
products, ratings_df = load_data()
predicted_df, item_sim_df = train_model(ratings_df)
init_cart()
users       = ratings_df.index.tolist()
product_ids = list(products.keys())


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">🛒 Smart Grocery Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Collaborative Filtering + Computer Vision — ML/CV Domain Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-number">{len(users)}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-number">{len(products)}</div><div class="stat-label">Products</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-number">SVD</div><div class="stat-label">CF Model</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-number">CV</div><div class="stat-label">Vision Module</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">📦 Product Catalog</div>', unsafe_allow_html=True)
    cats     = sorted(set(v["category"] for v in products.values()))
    sel_cats = st.multiselect("Filter by Category", cats, default=cats[:4])
    filtered = {pid: pdata for pid, pdata in products.items() if pdata["category"] in sel_cats}
    cols     = st.columns(4)
    for i, (pid, pdata) in enumerate(filtered.items()):
        with cols[i % 4]:
            badges = "".join([f'<span class="badge badge-green">{t}</span>' for t in pdata["tags"][:2]])
            st.markdown(f"""<div class="product-card">
                <span class="product-emoji">{pdata['emoji']}</span>
                <div class="product-name">{pdata['name']}</div>
                <div style="color:#e74c3c;font-weight:700;margin:0.25rem 0;">₹{pdata['price']}</div>
                <div>{badges}</div></div>""", unsafe_allow_html=True)
            if st.button("🛒 Add", key=f"home_cart_{pid}"):
                add_to_cart(pid, products)
                st.toast(f"✅ {pdata['name']} added!", icon="🛒")


# ═══════════════════════════════════════════════════════════════
# PAGE: CF RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
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
                        st.markdown(f"""<div class="product-card">
                            <span class="product-emoji">{p['emoji']}</span>
                            <div class="product-name">{p['name']}</div>
                            <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                            <div class="product-score">⭐ Score: {score:.2f}</div></div>""", unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cf_rec_{pid}_{u}"):
                            add_to_cart(pid, products)
                            st.toast(f"✅ {p['name']} added!", icon="🛒")

    with tab2:
        st.markdown('<div class="section-header">🔗 Item-Item Similarity</div>', unsafe_allow_html=True)
        sel_product = st.selectbox("Select a product", product_ids,
            format_func=lambda x: f"{products[x]['emoji']} {products[x]['name']}")
        if st.button("🔍 Find Similar Products"):
            base_cat = products[sel_product]["category"]
            allowed  = RELATED_CATEGORIES.get(base_cat, [base_cat])
            sim_pids, sim_scores = get_similar_products(
                sel_product, item_sim_df, products, n_recs, filter_categories=allowed
            )
            st.markdown(f'<div class="section-header">Products similar to {products[sel_product]["name"]}</div>', unsafe_allow_html=True)
            if not sim_pids:
                st.info("No similar products found.")
            scols = st.columns(3)
            for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                if pid not in products: continue
                p = products[pid]
                with scols[i % 3]:
                    st.markdown(f"""<div class="product-card">
                        <span class="product-emoji">{p['emoji']}</span>
                        <div class="product-name">{p['name']}</div>
                        <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                        <div class="product-score">🔗 Similarity: {score:.3f}</div></div>""", unsafe_allow_html=True)
                    if st.button("🛒 Add to Cart", key=f"sim_{pid}_{sel_product}"):
                        add_to_cart(pid, products)
                        st.toast(f"✅ {p['name']} added!", icon="🛒")


# ═══════════════════════════════════════════════════════════════
# PAGE: IMAGE SCANNER  (FULLY IMPROVED)
# ═══════════════════════════════════════════════════════════════
elif page == "📸 Image Scanner":
    st.markdown('<h1 class="main-title">📸 Product Image Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grocery photo → CV identifies it → Shows matched products + CF suggestions</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("📌 Upload any Indian grocery/household product image. AI identifies the product and suggests similar items using Collaborative Filtering.")

    up_col, prev_col = st.columns([1, 1])
    with up_col:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a grocery product image", type=["jpg","jpeg","png","webp"])

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
                st.session_state.pop("cv_done", None)
                st.session_state.pop("cv_tags", None)
                st.session_state.pop("cv_pids", None)
                st.rerun()

        # Manual search (always visible)
        st.markdown("---")
        st.markdown('<div class="section-header">🔎 Manual Product Search</div>', unsafe_allow_html=True)
        manual_col1, manual_col2 = st.columns([3, 1])
        with manual_col1:
            manual_query = st.text_input(
                "Type product name if AI misidentifies (e.g. butter, maggi, chips)",
                key="manual_search"
            )
        with manual_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            manual_btn = st.button("🔍 Search", use_container_width=True)

        if manual_btn and manual_query.strip():
            q = manual_query.strip().lower()
            manual_tags = [{"tag": normalize_tag(q), "confidence": 95}]
            for w in q.split():
                if len(w) >= 3:
                    manual_tags.append({"tag": w, "confidence": 75})
            matched_pids = find_products_from_tags(manual_tags, products)
            st.session_state["cv_tags"]        = manual_tags
            st.session_state["cv_pids"]        = matched_pids
            st.session_state["cv_method"]      = f"🔎 Manual Search: '{manual_query}'"
            st.session_state["cv_debug"]       = []
            st.session_state["cv_done"]        = True

        if analyze:
            pb = st.progress(0, text="🧠 Initializing AI Vision...")
            time.sleep(0.2)
            pb.progress(20, text="🤖 Sending image to Gemini...")
            tags, method, debug_log = classify_image(img_bytes)
            pb.progress(75, text="🔍 Matching products in catalog...")
            time.sleep(0.2)
            matched_pids = find_products_from_tags(tags, products)
            pb.progress(100, text="✅ Analysis complete!")
            time.sleep(0.3)
            pb.empty()

            st.session_state["cv_tags"]   = tags
            st.session_state["cv_pids"]   = matched_pids
            st.session_state["cv_method"] = method
            st.session_state["cv_debug"]  = debug_log
            st.session_state["cv_done"]   = True

    # ── Results section ──
    if st.session_state.get("cv_done"):
        method  = st.session_state.get("cv_method", "")
        tags    = st.session_state.get("cv_tags", [])
        matched = st.session_state.get("cv_pids", [])

        st.markdown("---")

        # Detection summary banner
        primary_tag = ""
        if tags:
            t0 = tags[0]
            primary_tag = t0.get("tag","") if isinstance(t0, dict) else str(t0)

        st.markdown(f"""<div class="detected-product-banner">
            <span style="font-size:1.5rem;">🔎</span>
            <span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#2ECC71;margin-left:0.5rem;">
                Detected: {primary_tag.title() if primary_tag else "Unknown"}
            </span>
            <span style="font-size:0.8rem;color:#aaa;margin-left:1rem;">via {method}</span>
        </div>""", unsafe_allow_html=True)

        # All detected tags
        st.markdown('<div class="section-header">🏷️ All Detected Labels</div>', unsafe_allow_html=True)
        tags_html = ""
        for item in tags[:10]:
            if isinstance(item, dict):
                tag_name = item.get("tag", "")
                conf     = item.get("confidence", 0)
                conf_int = min(int(conf), 100)
                tags_html += f"""<div style="margin:4px 0;">
                    <span class="badge badge-orange">{tag_name}</span>
                    <span class="badge badge-conf">{conf:.0f}%</span>
                    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf_int}%;"></div></div>
                </div>"""
            else:
                tags_html += f'<span class="badge badge-orange">{item}</span>'
        st.markdown(f'<div style="margin:0.5rem 0 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

        # Debug log
        if st.session_state.get("cv_debug"):
            with st.expander("🔧 API Debug Log"):
                for msg in st.session_state["cv_debug"]:
                    st.markdown(f"`{msg}`")

        # Matched products
        st.markdown('<div class="section-header">🛒 Matched Products in Catalog</div>', unsafe_allow_html=True)
        if not matched:
            top_tag = primary_tag or "this item"
            st.warning(f"⚠️ Detected **'{top_tag}'** but no catalog match found. Try manual search above with a more specific name.")
        else:
            m_cols = st.columns(min(len(matched), 3))
            for i, pid in enumerate(matched):
                p = products.get(pid)
                if not p: continue
                with m_cols[i % 3]:
                    cat_badge = f'<span class="badge badge-blue">{p["category"]}</span>'
                    tag_badges = "".join([f'<span class="badge badge-green">{t}</span>' for t in p["tags"][:2]])
                    st.markdown(f"""<div class="product-card">
                        <span class="product-emoji">{p['emoji']}</span>
                        <div class="product-name">{p['name']}</div>
                        <div style="color:#e74c3c;font-weight:700;margin:0.25rem 0;">₹{p['price']}</div>
                        {cat_badge}
                        <div style="margin-top:0.25rem;">{tag_badges}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("🛒 Add", key=f"cv_match_{pid}"):
                        add_to_cart(pid, products)
                        st.toast(f"✅ {p['name']} added!", icon="🛒")

        # CF-based "You might also like" — IMPROVED: seeds from all matched products
        if matched:
            st.markdown('<div class="section-header">🤖 You Might Also Like</div>', unsafe_allow_html=True)
            base_cat = products.get(matched[0], {}).get("category", "")
            st.markdown(f'<small style="color:#7f8c8d;">Based on detected category: <b>{base_cat}</b> · Powered by Collaborative Filtering</small>', unsafe_allow_html=True)

            sim_pids, sim_scores = get_smart_recommendations(matched, products, item_sim_df, n=n_recs)

            if not sim_pids:
                st.info("No CF suggestions found for this product.")
            else:
                s_cols = st.columns(4)
                for i, (pid, score) in enumerate(zip(sim_pids, sim_scores)):
                    if pid not in products: continue
                    p = products[pid]
                    with s_cols[i % 4]:
                        st.markdown(f"""<div class="product-card">
                            <span class="product-emoji">{p['emoji']}</span>
                            <div class="product-name">{p['name']}</div>
                            <div style="color:#e74c3c;font-weight:700;">₹{p['price']}</div>
                            <div class="product-score">🔗 {score:.3f}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cv_cf_{pid}_{i}"):
                            add_to_cart(pid, products)
                            st.toast(f"✅ {p['name']} added!", icon="🛒")


# ═══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════
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
    sample_cols = [c for c in ratings_df.columns[:10] if c in products]
    sample      = ratings_df.iloc[:15][sample_cols].copy()
    sample.columns = [f"{products[c]['emoji']}{products[c]['name'][:8]}" for c in sample.columns]
    st.dataframe(sample.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown('<div class="section-header">🔬 SVD Explained Variance</div>', unsafe_allow_html=True)
    n_comp   = min(10, ratings_df.shape[0] - 1, ratings_df.shape[1] - 1)
    svd_test = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_test.fit(ratings_df.values)
    var_df = pd.DataFrame({
        "Component":              [f"C{i+1}" for i in range(n_comp)],
        "Explained Variance (%)": (svd_test.explained_variance_ratio_ * 100).round(2)
    })
    st.bar_chart(var_df.set_index("Component"))
    st.caption(f"Total variance explained: {svd_test.explained_variance_ratio_.sum()*100:.1f}%")
