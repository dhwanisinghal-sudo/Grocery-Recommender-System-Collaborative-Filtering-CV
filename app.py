import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import requests
import re
import json
import time
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

# ══════════════════════════════════════════════════════════════
# CATEGORY CONFIG
# ══════════════════════════════════════════════════════════════
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
    "Dairy":         ["Dairy", "Bakery", "Beverages"],
    "Grains":        ["Grains", "Spices", "Condiments", "Noodles"],
    "Spices":        ["Spices", "Grains", "Condiments", "Noodles"],
    "Noodles":       ["Noodles", "Grains", "Spices", "Condiments"],
    "Condiments":    ["Condiments", "Spices", "Grains", "Noodles", "Bakery"],
    "Personal Care": ["Personal Care", "Home Care"],
    "Health":        ["Health", "Beverages", "Grains"],
    "Home Care":     ["Home Care", "Personal Care"],
    "Frozen":        ["Frozen", "Snacks", "Noodles"],
}

def get_emoji(category):
    return CATEGORY_EMOJI.get(category, "🛒")

# ══════════════════════════════════════════════════════════════
# KEYWORD → PRODUCT MAPPING  (exact PIDs from products.csv)
# ══════════════════════════════════════════════════════════════
GROCERY_KEYWORDS = {
    # ── DAIRY ──
    "butter":            ["P021","P135"],
    "amul butter":       ["P021"],
    "priya gold butter": ["P135"],
    "ghee":              ["P025"],
    "amul ghee":         ["P025"],
    "cheese":            ["P022"],
    "amul cheese":       ["P022"],
    "cheese slices":     ["P022"],
    "yogurt":            ["P023","P029","P030"],
    "curd":              ["P030","P023","P029"],
    "dahi":              ["P030"],
    "amul dahi":         ["P030"],
    "greek yogurt":      ["P029"],
    "epigamia":          ["P029"],
    "nestle yogurt":     ["P023"],
    "paneer":            ["P024"],
    "mother dairy paneer":["P024"],
    "cream":             ["P028"],
    "amul cream":        ["P028"],
    "lassi":             ["P136"],
    "amul lassi":        ["P136"],
    "shrikhand":         ["P026"],
    "amul shrikhand":    ["P026"],
    "condensed milk":    ["P027"],
    "milkmaid":          ["P027"],
    "milk":              ["P021","P030","P027","P136"],
    "dairy":             ["P021","P022","P023","P024","P025","P028","P030"],
    "amul":              ["P021","P022","P025","P026","P028","P030","P117","P136"],
    "amul kool":         ["P117"],
    "cold coffee":       ["P117"],

    # ── BAKERY ──
    "parle g":           ["P001"],
    "parle-g":           ["P001"],
    "parle":             ["P001","P010"],
    "good day":          ["P002"],
    "cashew biscuit":    ["P002"],
    "marie gold":        ["P003"],
    "britannia marie":   ["P003"],
    "bourbon":           ["P004"],
    "chocolate cream biscuit": ["P004"],
    "milk bikis":        ["P005"],
    "hide and seek":     ["P006"],
    "hide seek":         ["P006"],
    "digestive":         ["P007"],
    "nutrichoice":       ["P007"],
    "krackjack":         ["P008"],
    "monaco":            ["P009"],
    "tiger biscuit":     ["P010"],
    "glucose biscuit":   ["P001","P010"],
    "50 50":             ["P132"],
    "dark fantasy":      ["P133"],
    "dream cream":       ["P134"],
    "sunfeast":          ["P055","P134"],
    "butter bite":       ["P135"],
    "britannia":         ["P003","P080","P132","P133","P134"],
    "biscuit":           ["P001","P002","P003","P004","P005","P006","P007","P008","P009","P010"],
    "cookie":            ["P006","P133","P134"],
    "cracker":           ["P008","P009"],
    "bakery":            ["P001","P002","P003","P006","P007"],

    # ── SNACKS ──
    "lays classic":      ["P011"],
    "lays magic masala": ["P018"],
    "lays spanish":      ["P143"],
    "lays":              ["P011","P018","P143"],
    "kurkure":           ["P012"],
    "bingo":             ["P013"],
    "mad angles":        ["P013"],
    "pringles":          ["P014"],
    "aloo bhujia":       ["P015"],
    "haldiram bhujia":   ["P015"],
    "haldiram mixture":  ["P016"],
    "moong dal":         ["P017"],
    "haldiram moong":    ["P017"],
    "too yumm":          ["P019"],
    "bikano":            ["P020"],
    "chana chur":        ["P020"],
    "haldiram sev":      ["P141"],
    "sev":               ["P141"],
    "bikaji boondi":     ["P142"],
    "boondi":            ["P142"],
    "doritos":           ["P144"],
    "nacho":             ["P144"],
    "chips":             ["P011","P012","P013","P014","P018","P143","P144"],
    "namkeen":           ["P015","P016","P017","P020","P141","P142"],
    "bhujia":            ["P015"],
    "haldiram":          ["P015","P016","P017","P112","P141","P142"],
    "snack":             ["P011","P012","P013","P015","P016","P019","P020"],

    # ── NOODLES ──
    "maggi noodles":     ["P051","P131"],
    "maggi 2 minute":    ["P051"],
    "maggi atta":        ["P057"],
    "maggi masala":      ["P131"],
    "maggi sauce":       ["P072"],
    "maggi":             ["P051","P057","P131"],
    "yippee":            ["P052"],
    "magic masala":      ["P052"],
    "knorr soupy":       ["P053"],
    "soupy noodles":     ["P053"],
    "wai wai":           ["P054"],
    "top ramen":         ["P058"],
    "patanjali noodles": ["P059"],
    "sunfeast pasta":    ["P055"],
    "borges pasta":      ["P056"],
    "smith jones pasta": ["P060"],
    "pasta":             ["P055","P056","P060"],
    "noodles":           ["P051","P052","P053","P054","P057","P058","P059","P131"],
    "instant noodles":   ["P051","P052","P053","P054","P131"],
    "ramen":             ["P051","P052","P054","P131"],
    "vermicelli":        ["P052","P057"],

    # ── GRAINS ──
    "india gate":        ["P031"],
    "basmati rice":      ["P031"],
    "daawat":            ["P032"],
    "rozana rice":       ["P032"],
    "rice":              ["P031","P032"],
    "tata sampann toor": ["P033"],
    "toor dal":          ["P033"],
    "tata sampann chana":["P034"],
    "chana dal":         ["P034"],
    "tata sampann":      ["P033","P034"],
    "dal":               ["P033","P034"],
    "aashirvaad atta":   ["P035"],
    "aashirvaad":        ["P035"],
    "fortune atta":      ["P145"],
    "chakki atta":       ["P145"],
    "atta":              ["P035","P145"],
    "pillsbury maida":   ["P036"],
    "maida":             ["P036"],
    "flour":             ["P035","P036","P145"],
    "quaker oats":       ["P037"],
    "saffola oats":      ["P038"],
    "saffola masala oats":["P146"],
    "oats":              ["P037","P038","P094","P146"],
    "mtr poha":          ["P039"],
    "poha":              ["P039"],
    "suji":              ["P040"],
    "rawa":              ["P040"],
    "semolina":          ["P040"],
    "wheat":             ["P035","P145"],
    "grain":             ["P031","P032","P033","P034"],
    "lentil":            ["P033","P034"],
    "pulses":            ["P033","P034"],

    # ── SPICES ──
    "mdh garam masala":  ["P041"],
    "garam masala":      ["P041"],
    "mdh":               ["P041","P045","P047","P147"],
    "everest kitchen king":["P042"],
    "kitchen king":      ["P042"],
    "everest":           ["P042","P044","P046","P048","P148"],
    "turmeric":          ["P043"],
    "haldi":             ["P043"],
    "catch turmeric":    ["P043"],
    "red chilli":        ["P044"],
    "chilli powder":     ["P044"],
    "everest chilli":    ["P044"],
    "coriander powder":  ["P046"],
    "dhania":            ["P046"],
    "rajma masala":      ["P045"],
    "mdh rajma":         ["P045"],
    "chana masala":      ["P047"],
    "mdh chana":         ["P047"],
    "pav bhaji masala":  ["P048"],
    "everest pav bhaji": ["P048"],
    "biryani masala":    ["P147"],
    "mdh biryani":       ["P147"],
    "sabji masala":      ["P148"],
    "everest sabji":     ["P148"],
    "masala":            ["P041","P042","P045","P047","P048","P147","P148"],
    "spice":             ["P041","P042","P043","P044","P147","P148"],
    "saffola oil":       ["P049"],
    "fortune sunflower": ["P050"],
    "sunflower oil":     ["P050"],
    "cooking oil":       ["P049","P050"],
    "oil":               ["P049","P050"],

    # ── CONDIMENTS ──
    "kissan jam":        ["P071"],
    "mixed fruit jam":   ["P071"],
    "britannia jam":     ["P080"],
    "date fig jam":      ["P080"],
    "jam":               ["P071","P080"],
    "maggi hot sweet":   ["P072"],
    "hot sweet sauce":   ["P072"],
    "heinz ketchup":     ["P073"],
    "tomato ketchup":    ["P073"],
    "ketchup":           ["P073"],
    "chings schezwan":   ["P074"],
    "schezwan chutney":  ["P074"],
    "chutney":           ["P074"],
    "veeba burger sauce":["P075"],
    "burger sauce":      ["P075"],
    "sauce":             ["P072","P073","P075"],
    "dr oetker mayo":    ["P076"],
    "mayonnaise":        ["P076"],
    "mayo":              ["P076"],
    "druk honey":        ["P077"],
    "dabur honey":       ["P096"],
    "honey":             ["P077","P096"],
    "nutella":           ["P078"],
    "chocolate spread":  ["P078"],
    "amul peanut butter":["P079"],
    "peanut butter":     ["P079"],
    "spread":            ["P078","P079"],
    "condiment":         ["P071","P072","P073","P074","P076","P077","P078","P079"],

    # ── DRINKS ──
    "tropicana orange":  ["P061"],
    "tropicana guava":   ["P138"],
    "tropicana":         ["P061","P138"],
    "real mango":        ["P062"],
    "real pomegranate":  ["P137"],
    "real juice":        ["P062","P068","P137","P138"],
    "frooti":            ["P063"],
    "mango drink":       ["P062","P063","P066"],
    "maaza":             ["P066"],
    "paper boat":        ["P067"],
    "aamras":            ["P067"],
    "b natural":         ["P068"],
    "mixed fruit juice": ["P068"],
    "sting":             ["P064"],
    "red bull":          ["P070"],
    "energy drink":      ["P064","P070"],
    "limca":             ["P065"],
    "7up":               ["P140"],
    "appy fizz":         ["P139"],
    "soda":              ["P065","P139","P140"],
    "bisleri":           ["P069"],
    "water":             ["P069"],
    "orange juice":      ["P061"],
    "mango juice":       ["P062","P063","P066"],
    "juice":             ["P061","P062","P063","P066","P067","P068","P137","P138"],
    "fruit":             ["P061","P062","P063","P066","P137","P138"],
    "drink":             ["P061","P062","P063","P064","P066"],

    # ── BEVERAGES ──
    "tata tea gold":     ["P121"],
    "tata tea":          ["P121"],
    "red label":         ["P122"],
    "nescafe":           ["P123"],
    "nescafe classic":   ["P123"],
    "bru gold":          ["P124"],
    "bru":               ["P124"],
    "lipton green tea":  ["P125"],
    "green tea":         ["P125"],
    "tetley masala":     ["P126"],
    "masala chai":       ["P126"],
    "davidoff":          ["P127"],
    "bournvita":         ["P128"],
    "milo":              ["P129"],
    "taj mahal tea":     ["P130"],
    "tea":               ["P121","P122","P125","P126","P130"],
    "coffee":            ["P123","P124","P127"],
    "health drink":      ["P091","P092","P093","P128","P129"],
    "beverage":          ["P121","P122","P123","P124","P125"],
    "chai":              ["P121","P122","P126","P130"],

    # ── PERSONAL CARE ──
    "colgate":           ["P081"],
    "toothpaste":        ["P081"],
    "oral b":            ["P082"],
    "toothbrush":        ["P082"],
    "dove soap":         ["P083"],
    "dettol soap":       ["P084"],
    "pears soap":        ["P149"],
    "soap":              ["P083","P084","P149"],
    "head shoulders":    ["P085"],
    "pantene":           ["P086"],
    "shampoo":           ["P085","P086"],
    "nivea lotion":      ["P087"],
    "body lotion":       ["P087"],
    "lotion":            ["P087"],
    "parachute":         ["P088"],
    "coconut oil":       ["P088"],
    "gillette":          ["P089"],
    "razor":             ["P089"],
    "whisper":           ["P090"],
    "garnier micellar":  ["P150"],
    "micellar water":    ["P150"],
    "face wash":         ["P150"],
    "personal care":     ["P081","P083","P084","P085","P087"],

    # ── HOME CARE ──
    "surf excel":        ["P101"],
    "ariel":             ["P102"],
    "detergent":         ["P101","P102"],
    "vim":               ["P103"],
    "dishwash":          ["P103"],
    "harpic":            ["P104"],
    "toilet cleaner":    ["P104"],
    "colin":             ["P105"],
    "glass cleaner":     ["P105"],
    "lizol":             ["P106"],
    "floor cleaner":     ["P106"],
    "odonil":            ["P107"],
    "room freshener":    ["P107"],
    "scotch brite":      ["P108"],
    "scrub":             ["P108"],
    "mortein":           ["P109"],
    "good knight":       ["P110"],
    "mosquito":          ["P109","P110"],
    "cleaner":           ["P101","P103","P105","P106"],

    # ── HEALTH ──
    "horlicks":          ["P091"],
    "complan":           ["P092"],
    "ensure":            ["P093"],
    "saffola muesli":    ["P094"],
    "muesli":            ["P094"],
    "patanjali chyawanprash": ["P095"],
    "himalaya chyawanprash": ["P100"],
    "chyawanprash":      ["P095","P100"],
    "dabur":             ["P096"],
    "revital":           ["P097"],
    "vitamin":           ["P097"],
    "pediasure":         ["P098"],
    "glucon d":          ["P099"],
    "glucon":            ["P099"],
    "patanjali":         ["P095","P059","P100"],
    "himalaya":          ["P100"],
    "supplement":        ["P091","P092","P093","P098"],
    "immunity":          ["P095","P096","P100"],
    "ayurvedic":         ["P095","P100"],

    # ── FROZEN ──
    "mccain":            ["P111"],
    "mccain fries":      ["P111"],
    "smiles fries":      ["P111"],
    "itc fries":         ["P118"],
    "farmland fries":    ["P118"],
    "fries":             ["P111","P118"],
    "french fries":      ["P111","P118"],
    "haldiram dal makhani": ["P112"],
    "dal makhani":       ["P112"],
    "mtr paneer":        ["P113"],
    "paneer butter masala": ["P113"],
    "gits gulab jamun":  ["P114"],
    "gulab jamun":       ["P114"],
    "mother dairy ice cream": ["P115"],
    "cornetto":          ["P116"],
    "kwality walls":     ["P116"],
    "vadilal kulfi":     ["P119"],
    "mango kulfi":       ["P119"],
    "kulfi":             ["P119"],
    "ice cream":         ["P115","P116","P119"],
    "sumeru cutlets":    ["P120"],
    "cutlets":           ["P120"],
    "frozen":            ["P111","P112","P113","P114","P115","P116"],
    "ready meal":        ["P112","P113"],
    "ready to eat":      ["P112","P113","P114"],

    # ── GENERIC VISUAL FALLBACKS ──
    "packet":            ["P051","P011","P001","P021"],
    "packaged food":     ["P051","P011","P001"],
    "bottle":            ["P049","P050","P061","P069"],
    "can":               ["P061","P063","P064"],
    "box":               ["P001","P031","P091"],
    "jar":               ["P071","P077","P078","P079"],
    "pouch":             ["P051","P052","P035"],
    "tube":              ["P081"],
    "container":         ["P023","P025","P028"],
    "yellow":            ["P025","P051","P062","P063"],
    "white":             ["P021","P024","P030","P035"],
    "red":               ["P044","P064","P073","P122"],
    "green":             ["P033","P046","P049","P125"],
    "brown":             ["P004","P006","P025","P133"],
    "golden":            ["P025","P077"],
    "block":             ["P021","P022","P024"],
    "slab":              ["P021","P022","P024"],
    "foil":              ["P021","P025"],
    "wrapped":           ["P021","P025"],
    "margarine":         ["P021"],
    "food":              ["P001","P011","P021","P031","P051","P061"],
    "grocery":           ["P001","P011","P021","P031","P051","P061"],
    "indian food":       ["P033","P035","P041","P051","P121"],
    "cooking":           ["P025","P033","P035","P041","P049","P050"],
    "breakfast":         ["P037","P038","P039","P094","P121","P122"],
    "kids":              ["P001","P005","P010","P051","P063","P091"],
    "sweet":             ["P006","P026","P077","P078","P114"],
    "chocolate":         ["P004","P006","P078","P092","P128","P133"],
    "spicy":             ["P012","P013","P015","P041","P044"],
    "healthy":           ["P007","P029","P037","P091","P094","P096"],
    "organic":           ["P033","P035","P037","P095"],
    "band":              ["P021","P025","P030"],
    "band aid":          ["P021","P025","P030"],
    "adhesive":          ["P021","P025","P030"],
    "wrapper":           ["P051","P011","P001"],
    "rectangular":       ["P021","P022","P024"],
    "solid":             ["P021","P022","P024","P025"],
}

LOW_PRIORITY_TAGS = {
    "food", "bottle", "yellow", "ripe", "grain", "cereal", "beverage",
    "drink", "juice", "sweet", "grocery", "groceries", "food item",
    "indian", "staple", "packaged", "packaged food", "processed food",
    "band", "wrapper", "label", "block", "slab", "rectangular",
    "foil", "wrapped", "golden", "solid", "white", "red", "green",
    "brown", "container", "box", "can", "bottle"
}


def normalize_tag(tag: str) -> str:
    return " ".join(tag.lower().strip().split())


# ══════════════════════════════════════════════════════════════
# CONFLICT-RESOLUTION GROUPS
# Vision models often confuse visually-similar dairy/snack items
# (e.g. butter block vs cream pouch vs yogurt cup). Each group below
# lists "anchor" tags that, if spotted with decent confidence, should
# win over the other conflicting tags in the SAME group rather than
# letting both sets of products get merged together.
# Format: anchor_tag -> {"pids": [...], "suppress": {other tags in group}}
# ══════════════════════════════════════════════════════════════
DAIRY_SOLID_VS_LIQUID_GROUP = {"butter", "ghee", "paneer", "cheese", "yogurt", "curd", "dahi", "cream", "lassi", "milk"}

CONFLICT_OVERRIDES = {
    "butter": {"pids": ["P021", "P135"], "suppress": {"cream", "yogurt", "curd", "dahi", "lassi"}},
    "ghee":   {"pids": ["P025"],          "suppress": {"cream", "yogurt", "curd", "dahi", "butter"}},
    "paneer": {"pids": ["P024"],          "suppress": {"cream", "yogurt", "curd", "dahi", "butter", "cheese"}},
    "cheese": {"pids": ["P022"],          "suppress": {"cream", "yogurt", "curd", "dahi", "paneer"}},
}


def apply_conflict_resolution(raw_tag_texts):
    """
    Looks at all detected tags. If a strong 'anchor' tag (butter, ghee,
    paneer, cheese) is present, suppress the visually-confusable tags
    in the same dairy group so they don't dilute the match, and force-add
    the anchor's correct product IDs.
    """
    forced_pids = set()
    suppressed_tags = set()

    for anchor, cfg in CONFLICT_OVERRIDES.items():
        if any(anchor == t or anchor in t for t in raw_tag_texts):
            forced_pids.update(cfg["pids"])
            suppressed_tags.update(cfg["suppress"])

    cleaned_tags = [t for t in raw_tag_texts if t not in suppressed_tags]
    return cleaned_tags, forced_pids


def find_products_from_tags(tag_dicts, products):
    matched_high = set()
    matched_low  = set()

    raw_tag_texts = []
    confidences   = []
    for item in tag_dicts:
        if isinstance(item, dict):
            raw_tag_texts.append(normalize_tag(str(item.get("tag", ""))))
            confidences.append(float(item.get("confidence", 0) or 0))
        else:
            raw_tag_texts.append(normalize_tag(str(item)))
            confidences.append(0.0)

    # ── NOT IN CATALOG: fresh produce / non-packaged items ──
    # Case 1: Gemini explicitly returned the not_in_catalog tag.
    # Case 2: Gemini still guessed a packaged-product tag, but with
    #         very low confidence (e.g. 3-5%) — this is the model
    #         essentially saying "I'm not sure this is even a real
    #         match", which happens with fruits/veggies/unknown items.
    #         Treat low-confidence top guesses the same way rather
    #         than forcing a random product onto the user.
    explicit_not_in_catalog = any(
        t in ("not in catalog", "not_in_catalog", "notincatalog") for t in raw_tag_texts
    )
    top_confidence = max(confidences) if confidences else 0
    low_confidence_guess = len(confidences) > 0 and top_confidence < 20

    if explicit_not_in_catalog or low_confidence_guess:
        return []

    cleaned_tags, forced_pids = apply_conflict_resolution(raw_tag_texts)
    for pid in forced_pids:
        if pid in products:
            matched_high.add(pid)

    for tag in cleaned_tags:
        tag_words = [w for w in tag.split() if len(w) >= 3]

        for keyword, pids in GROCERY_KEYWORDS.items():
            kw       = normalize_tag(keyword)
            kw_words = [w for w in kw.split() if len(w) >= 3]

            exact      = (kw == tag)
            substr     = (kw in tag) or (tag in kw)
            word_match = any(
                (tw in kw) or any(kw_w in tw for kw_w in kw_words)
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

    if not combined:
        fallback = ["P051","P011","P001","P021","P033","P061"]
        combined = [p for p in fallback if p in products]

    return combined[:6]


# ══════════════════════════════════════════════════════════════
# GEMINI PROMPT  — generic product-TYPE focused (brand-agnostic)
# Works for ANY image of a product type, not just our exact catalog
# brands — confidence reflects how sure Gemini is about the TYPE,
# not whether the exact brand matches our store.
# ══════════════════════════════════════════════════════════════
GEMINI_PROMPT = """You are an expert grocery product identifier. Your job is to identify the TYPE of grocery/food/household product shown, regardless of brand.

IMPORTANT: The exact brand in the image does not need to match any specific store's catalog. Focus on identifying what KIND of product it is (butter, chips, biscuit, juice, etc.) — any brand of that product type counts as a correct identification. Give HIGH confidence (70-95) whenever you can clearly tell the product type, even if you don't recognize the specific brand.

DAIRY IDENTIFICATION — judge by packaging shape/texture, not colour alone:
- BUTTER → solid, dense, rectangular block; wrapped in foil or waxed paper. Any brand (Amul, Britannia, Land O'Lakes, generic, etc.) → say "butter".
- GHEE → liquid/semi-solid in a jar/tin/pouch, golden, glossy/oily. → say "ghee".
- PANEER → soft white solid block, usually clear vacuum-sealed plastic (not foil). → say "paneer".
- CHEESE → individually wrapped slices, a firm block, or a wedge with a cheese-brand wrapper. → say "cheese".
- YOGURT/CURD/DAHI → semi-liquid in a cup, pot, or tub with a peel-off foil lid — never a solid block. → say "yogurt" or "curd".
- CREAM → pourable liquid in a small sealed pouch or carton. → say "cream".
- MILK → liquid in a pouch, tetra-pack, carton, or bottle. → say "milk".

KEY RULE: a SOLID block (not liquid, not in a cup/tub) wrapped in foil or paper is BUTTER, regardless of brand or exact colour shade. Liquid/semi-liquid items in cups, tubs, or pouches are yogurt/curd/cream — never call those "butter", and never call a solid block "yogurt" or "cream".

SNACKS (any brand counts):
- Thin, flat, fried, crinkled discs/strips in a packet = chips
- Fried salty mixture (small irregular pieces) in a packet = namkeen/mixture
- Thin fried strips = sev or bhujia
- Triangular fried crisp = nacho-style chip
- Baked, flat, uniform shape in a box/packet = biscuit/cookie (NOT chips)

NOODLES & PASTA (any brand counts):
- Square/rectangular dried noodle cake in a wrapper = instant noodles
- Long thin dried strands in a box/bag = pasta or spaghetti

SPICES (any brand counts):
- Small packet/jar/box of fine powder = spice or masala
- Yellow powder = turmeric
- Red powder = chilli powder
- Multi-ingredient spice blend = masala

BEVERAGES (any brand counts):
- Carbonated drink in a bottle/can = soda
- Fruit-coloured liquid in a bottle/carton = juice
- Leaves/bags/powder for hot drinks = tea or coffee

GENERAL RULE: identify the most likely PRODUCT TYPE even if the exact brand is unfamiliar to you. A product type you can clearly recognize (e.g. "this is clearly a chips packet" or "this is clearly a butter block") should get confidence 70+ even without knowing the brand. Only use confidence below 30 when the product type itself is genuinely ambiguous or unclear.

OUR STORE ONLY SELLS PACKAGED GROCERY ITEMS — no loose fresh fruits, vegetables, salads, meat, or non-packaged produce.
If the image shows a fresh, unpackaged fruit, vegetable, salad, raw meat, or any item that is clearly NOT a packaged product (no wrapper, no box, no bottle, no jar, no branded label of any kind), return exactly this single tag instead of guessing:
[{"tag": "not_in_catalog", "confidence": 99.0}]

Return ONLY a valid JSON array with 4-6 tags, most specific first (or the single not_in_catalog tag if applicable).
Format: [{"tag": "product name", "confidence": 85.0}, ...]
Rules:
- First tag: most specific guess (brand name if you recognize it, e.g. "amul butter")
- Then the generic product type (e.g. "butter") — THIS is the most important tag, always include it even if you don't know the brand
- Then the category (e.g. "dairy")
- Confidence 0-100, calibrated to how sure you are about the PRODUCT TYPE, not the brand
- NO markdown, NO extra text, ONLY the JSON array
"""


# ══════════════════════════════════════════════════════════════
# HF LABEL → GROCERY TAG MAP
# ══════════════════════════════════════════════════════════════
HF_LABEL_MAP = {
    # Dairy
    "butter": "butter", "ghee": "ghee", "paneer": "paneer",
    "cheese": "cheese", "yogurt": "yogurt", "curd": "curd",
    "cream": "cream", "milk": "milk", "lassi": "lassi",
    "milk can": "milk", "milk bottle": "milk", "dairy": "dairy",
    "condensed milk": "condensed milk", "shrikhand": "shrikhand",
    # Bakery
    "biscuit": "biscuit", "cookie": "biscuit", "cracker": "biscuit",
    "wafer": "biscuit", "bread": "bread", "loaf": "bread",
    "bagel": "bread", "toast": "bread", "cake": "bakery",
    "digestive": "digestive",
    # Snacks
    "chips": "chips", "potato chips": "chips", "crisps": "chips",
    "popcorn": "snack", "nacho": "nacho", "nachos": "nacho",
    "namkeen": "namkeen", "bhujia": "bhujia", "sev": "sev",
    "french fries": "fries", "fries": "fries", "snack": "snack",
    # Noodles
    "noodle": "noodles", "noodles": "noodles", "ramen": "ramen",
    "pasta": "pasta", "spaghetti": "noodles", "vermicelli": "noodles",
    "macaroni": "pasta", "instant noodles": "maggi noodles",
    # Grains
    "rice": "rice", "basmati": "basmati rice", "dal": "dal",
    "lentil": "dal", "lentils": "dal", "flour": "atta",
    "wheat flour": "atta", "wheat": "wheat", "atta": "atta",
    "oat": "oats", "oats": "oats", "oatmeal": "oats",
    "semolina": "suji", "poha": "poha",
    # Spices
    "masala": "masala", "spice": "spice", "spices": "spice",
    "turmeric": "turmeric", "chilli": "red chilli", "chili": "red chilli",
    "oil": "cooking oil", "cooking oil": "cooking oil",
    "sunflower oil": "sunflower oil", "coriander": "coriander powder",
    # Condiments
    "sauce": "sauce", "ketchup": "ketchup", "chutney": "chutney",
    "mayo": "mayonnaise", "mayonnaise": "mayonnaise",
    "honey": "honey", "jam": "jam", "peanut butter": "peanut butter",
    "nutella": "nutella", "spread": "spread",
    # Drinks
    "juice": "juice", "mango": "mango juice", "orange juice": "orange juice",
    "energy drink": "energy drink", "soda": "soda", "water": "water",
    "smoothie": "juice", "coconut water": "drink",
    # Beverages
    "tea": "tea", "green tea": "green tea", "coffee": "coffee",
    "bournvita": "bournvita", "horlicks": "horlicks", "milo": "milo",
    "chai": "chai",
    # Personal Care
    "soap": "soap", "shampoo": "shampoo", "lotion": "lotion",
    "toothpaste": "toothpaste", "toothbrush": "toothbrush",
    "face wash": "face wash", "moisturizer": "lotion",
    "razor": "razor", "coconut oil": "coconut oil",
    # Home Care
    "detergent": "detergent", "dishwash": "dishwash",
    "cleaner": "cleaner", "floor cleaner": "floor cleaner",
    "toilet cleaner": "toilet cleaner", "mosquito": "mosquito",
    # Health
    "supplement": "supplement", "vitamin": "vitamin",
    "muesli": "muesli", "chyawanprash": "chyawanprash",
    # Frozen
    "ice cream": "ice cream", "kulfi": "kulfi",
    "frozen": "frozen", "fries": "fries",
    # HF garbage labels → sensible mapping
    "band aid": "butter", "band": "butter", "bandage": "butter",
    "adhesive bandage": "butter", "adhesive": "butter",
    "packet": "packaged food", "package": "packaged food",
    "wrapper": "packaged food", "envelope": "packaged food",
    "sachet": "packaged food", "pouch": "packaged food",
    "bottle": "bottle", "can": "drink", "tin": "drink",
    "jar": "jar", "container": "container", "tube": "toothpaste",
    "box": "box", "carton": "box", "bag": "packaged food",
    "yellow": "ghee", "golden": "ghee", "white": "milk",
    "red": "red chilli", "green": "coriander powder",
    "orange": "orange juice", "brown": "biscuit",
    "block": "butter", "slab": "butter", "rectangular": "butter",
    "foil": "butter", "wrapped": "butter", "margarine": "butter",
    "fat": "cooking oil", "grain": "grain", "cereal": "oats",
    "food": "food", "vegetable": "vegetable", "fruit": "fruit",
}


def classify_image_with_hf(image_bytes):
    debug_messages = []

    # ── 1. GEMINI (primary) ──────────────────────────────────
    try:
        gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if not gemini_key:
            debug_messages.append("⚠️ GEMINI_API_KEY missing in secrets.toml")
        else:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]

            for gmodel in GEMINI_MODELS:
                API_URL = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{gmodel}:generateContent?key={gemini_key}"
                )
                payload = {
                    "contents": [{
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                            {"text": GEMINI_PROMPT}
                        ]
                    }]
                }
                try:
                    resp = requests.post(API_URL, json=payload, timeout=30)
                    if resp.status_code == 200:
                        data       = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            debug_messages.append(f"⚠️ {gmodel}: empty candidates")
                            continue
                        finish = candidates[0].get("finishReason", "")
                        if finish in ("SAFETY", "RECITATION"):
                            debug_messages.append(f"⚠️ {gmodel}: blocked ({finish})")
                            continue
                        text = candidates[0]["content"]["parts"][0]["text"].strip()
                        text = text.replace("```json", "").replace("```", "").strip()
                        s = text.find("["); e = text.rfind("]") + 1
                        if s != -1 and e > s:
                            text = text[s:e]
                        result = json.loads(text)
                        if result and isinstance(result, list):
                            for r in result:
                                if isinstance(r, dict) and "tag" in r:
                                    r["tag"] = normalize_tag(str(r["tag"]))
                            debug_messages.append(f"✅ Gemini ({gmodel}) success")
                            st.session_state["cv_debug"] = debug_messages
                            return result, None
                    elif resp.status_code == 429:
                        debug_messages.append(f"⚠️ {gmodel}: rate limit, trying next...")
                        continue
                    elif resp.status_code == 400:
                        msg = resp.json().get("error", {}).get("message", "")
                        debug_messages.append(f"❌ {gmodel}: bad request — {msg}")
                        break
                    else:
                        debug_messages.append(f"❌ {gmodel}: HTTP {resp.status_code}")
                        break
                except json.JSONDecodeError as je:
                    debug_messages.append(f"⚠️ {gmodel}: JSON parse error — {je}")
                    continue
                except Exception as e:
                    debug_messages.append(f"⚠️ {gmodel}: {str(e)[:80]}")
                    continue
    except Exception as e:
        debug_messages.append(f"❌ Gemini setup error: {str(e)[:80]}")

    # ── 2. HUGGING FACE (fallback) ───────────────────────────
    try:
        hf_token = str(st.secrets.get("HF_API_TOKEN", "")).strip()
        if not hf_token:
            debug_messages.append("⚠️ HF_API_TOKEN missing in secrets.toml")
        else:
            headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "image/jpeg"}
            HF_MODELS = [
                "nateraw/food",
                "Kaludi/grocery-products",
                "google/vit-large-patch16-224",
                "microsoft/resnet-50",
            ]
            for model in HF_MODELS:
                API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
                try:
                    resp = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
                    if resp.status_code == 200:
                        results = resp.json()
                        if isinstance(results, list) and results:
                            tags = []
                            seen = set()
                            for item in results[:10]:
                                raw   = item.get("label", "").lower().strip()
                                conf  = round(item.get("score", 0.0) * 100, 1)
                                if conf < 3:
                                    continue
                                raw = re.sub(r"\(.*?\)", "", raw).strip()
                                raw = raw.split(",")[0].strip()
                                raw = raw.split("/")[0].strip()

                                # Map to grocery tag
                                gtag = None
                                if raw in HF_LABEL_MAP:
                                    gtag = HF_LABEL_MAP[raw]
                                if not gtag:
                                    for key, val in HF_LABEL_MAP.items():
                                        if key in raw and len(key) >= 4:
                                            gtag = val; break
                                if not gtag and len(raw) >= 4:
                                    for key, val in HF_LABEL_MAP.items():
                                        if raw in key:
                                            gtag = val; break
                                if not gtag:
                                    gtag = raw.split(" ")[0]

                                gtag = normalize_tag(str(gtag))
                                if gtag and len(gtag) >= 3 and not gtag.isdigit() and gtag not in seen:
                                    seen.add(gtag)
                                    tags.append({"tag": gtag, "confidence": conf})

                            if tags:
                                debug_messages.append(f"✅ HF ({model}) — {len(tags)} tags")
                                st.session_state["cv_debug"] = debug_messages
                                return tags, None
                            else:
                                debug_messages.append(f"⚠️ HF ({model}): no usable tags")
                    elif resp.status_code == 503:
                        debug_messages.append(f"⚠️ HF ({model}): loading, skip")
                        continue
                    elif resp.status_code == 429:
                        debug_messages.append(f"⚠️ HF ({model}): rate limit, skip")
                        continue
                    else:
                        debug_messages.append(f"❌ HF ({model}): HTTP {resp.status_code}")
                        continue
                except Exception as e:
                    debug_messages.append(f"⚠️ HF ({model}): {str(e)[:60]}")
                    continue
    except Exception as e:
        debug_messages.append(f"❌ HF setup error: {str(e)[:80]}")

    st.session_state["cv_debug"] = debug_messages
    return None, "Both Gemini and HF Vision failed"


def fallback_color_analysis(image: Image.Image):
    img_small = image.resize((100, 100)).convert("RGB")
    pixels    = np.array(img_small).reshape(-1, 3).astype(float)
    nw_mask   = ~((pixels[:,0]>220)&(pixels[:,1]>220)&(pixels[:,2]>220))
    fg        = pixels[nw_mask] if nw_mask.sum() >= 50 else pixels
    r, g, b   = fg.mean(axis=0)
    bright    = (r + g + b) / 3

    if r>160 and g>130 and b<110 and r>b*1.7:
        return [{"tag":"banana","confidence":74},{"tag":"fruit","confidence":70}]
    elif r>190 and g>90 and g<170 and b<90 and r>g*1.2:
        return [{"tag":"mango juice","confidence":70},{"tag":"juice","confidence":65}]
    elif g>r and g>b and g>100:
        return [{"tag":"vegetable","confidence":70},{"tag":"coriander powder","confidence":65}]
    elif r>g*1.4 and r>b*1.4 and r>140:
        return [{"tag":"masala","confidence":67},{"tag":"red chilli","confidence":62}]
    elif bright>215 and r>200 and g>200 and b>200:
        return [{"tag":"butter","confidence":70},{"tag":"dairy","confidence":65}]
    elif bright<80:
        return [{"tag":"coffee","confidence":68},{"tag":"tea","confidence":65}]
    elif r>130 and g>90 and b<90 and r>g:
        return [{"tag":"biscuit","confidence":66},{"tag":"bakery","confidence":63}]
    elif b>r*1.1 and b>g*1.1:
        return [{"tag":"milk","confidence":66},{"tag":"dairy","confidence":63}]
    else:
        return [{"tag":"snack","confidence":63},{"tag":"packaged food","confidence":60}]


# ══════════════════════════════════════════════════════════════
# DATA LOADING & ML MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=0)
def load_data():
    import os
    products_df = None
    for path in ["data/products.csv", "./data/products.csv", "products.csv"]:
        if os.path.exists(path):
            products_df = pd.read_csv(path); break
    if products_df is None:
        st.error("❌ products.csv not found!"); st.stop()

    products = {}
    for _, row in products_df.iterrows():
        pid   = row["product_id"]
        tags  = [t.strip() for t in str(row.get("tags","")).split(",") if t.strip()]
        emoji = str(row.get("emoji","")).strip()
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
        for p in ["data/users_new.csv","./data/users_new.csv","users_new.csv"]:
            if os.path.exists(p):
                user_ids += pd.read_csv(p)["user_id"].tolist(); break
    except Exception:
        pass

    product_ids = list(products.keys())
    ratings_path = next((p for p in ["data/ratings.csv","./data/ratings.csv","ratings.csv"] if os.path.exists(p)), None)
    try:
        if not ratings_path: raise FileNotFoundError
        raw_df = pd.read_csv(ratings_path)
        matrix = raw_df.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="mean")
        matrix = matrix.reindex(index=user_ids, columns=product_ids, fill_value=0).fillna(0)
    except FileNotFoundError:
        np.random.seed(42)
        raw = np.random.choice([0,0,0,1,2,3,4,5], size=(len(user_ids), len(product_ids)),
                               p=[0.5,0.1,0.1,0.1,0.08,0.06,0.04,0.02])
        matrix = pd.DataFrame(raw, index=user_ids, columns=product_ids)

    return products, matrix


@st.cache_resource
def train_model(_df):
    n = min(20, _df.shape[0]-1, _df.shape[1]-1)
    svd          = TruncatedSVD(n_components=n, random_state=42)
    uf           = svd.fit_transform(_df.values)
    predicted    = np.dot(uf, svd.components_)
    predicted_df = pd.DataFrame(predicted, index=_df.index, columns=_df.columns)
    item_sim     = cosine_similarity(svd.components_.T)
    item_sim_df  = pd.DataFrame(item_sim, index=_df.columns, columns=_df.columns)
    return predicted_df, item_sim_df


def get_user_recommendations(user_id, df, predicted_df, n=6):
    bought = df.loc[user_id][df.loc[user_id] > 0].index.tolist()
    preds  = predicted_df.loc[user_id].copy()
    preds[bought] = -999
    return preds.nlargest(n).index.tolist(), bought


def get_similar_products(product_id, item_sim_df, products, n=5, filter_categories=None):
    if product_id not in item_sim_df.columns:
        return [], []
    sims = item_sim_df[product_id].drop(product_id).sort_values(ascending=False)
    if filter_categories:
        allowed = [p for p in sims.index if products.get(p,{}).get("category") in filter_categories]
        sims    = sims[allowed]
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
        st.session_state["cart"][pid] = {"name":p["name"],"price":p["price"],"emoji":p["emoji"],"qty":1}

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
        ca, cb = st.columns([3,1])
        with ca: st.markdown(f'<div style="font-size:0.8rem;color:#ecf0f1;">{item["emoji"]} {item["name"]} ×{item["qty"]}</div>', unsafe_allow_html=True)
        with cb: st.markdown(f'<div style="font-size:0.8rem;color:#2ECC71;">₹{item["price"]*item["qty"]}</div>', unsafe_allow_html=True)
        total += item["price"] * item["qty"]
    st.markdown(f'<div style="background:#2ECC71;border-radius:8px;padding:0.5rem;text-align:center;color:white;font-weight:700;margin-top:0.5rem;">Total: ₹{total}</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Cart"):
        st.session_state["cart"] = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛒 GrocerAI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["🏠 Home Dashboard","🤖 CF Recommendations","📸 Image Scanner","📊 Analytics"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_recs = st.slider("Number of Recommendations", 3, 10, 6)
    render_cart_sidebar()
    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;opacity:0.5;text-align:center;">Built with ❤️ using Streamlit<br>ML + CV Domain Project</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
products, ratings_df = load_data()
predicted_df, item_sim_df = train_model(ratings_df)
init_cart()
users       = ratings_df.index.tolist()
product_ids = list(products.keys())
n_users     = len(users)
n_products  = len(products)


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">🛒 Smart Grocery Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Collaborative Filtering + Computer Vision — ML/CV Domain Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-number">{n_users}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-number">{n_products}</div><div class="stat-label">Products</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-number">SVD</div><div class="stat-label">CF Model</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-number">CV</div><div class="stat-label">Vision Module</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">📦 Product Catalog</div>', unsafe_allow_html=True)
    cats     = sorted(set(v["category"] for v in products.values()))
    sel_cats = st.multiselect("Filter by Category", cats, default=cats[:4])
    filtered = {pid:pd for pid,pd in products.items() if pd["category"] in sel_cats}
    cols = st.columns(4)
    for i,(pid,pdata) in enumerate(filtered.items()):
        with cols[i%4]:
            badges = "".join([f'<span class="badge badge-green">{t}</span>' for t in pdata["tags"][:2]])
            st.markdown(f'<div class="product-card"><span class="product-emoji">{pdata["emoji"]}</span><div class="product-name">{pdata["name"]}</div><div style="color:#e74c3c;font-weight:700;margin:0.25rem 0;">₹{pdata["price"]}</div><div>{badges}</div></div>', unsafe_allow_html=True)
            if st.button("🛒 Add", key=f"home_{pid}"):
                add_to_cart(pid, products)
                st.toast(f"✅ {pdata['name']} added!", icon="🛒")


# ══════════════════════════════════════════════════════════════
# PAGE: CF RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "🤖 CF Recommendations":
    st.markdown('<h1 class="main-title">🤖 Collaborative Filtering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">SVD-based Matrix Factorization • Cosine Similarity</p>', unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs(["👤 User-Based Recommendations","🔗 Item Similarity"])

    with tab1:
        col1, col2 = st.columns([1,2])
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
                bought_html = "".join([f'<span class="badge badge-blue">{products[p]["emoji"]} {products[p]["name"]}</span>' for p in bought[:8] if p in products])
                st.markdown(f'<div style="margin-bottom:1rem;">{bought_html or "<i style=color:#aaa>No purchases yet</i>"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-header">🎁 Recommended for {u}</div>', unsafe_allow_html=True)
                rcols = st.columns(3)
                for i, pid in enumerate(recs):
                    if pid not in products: continue
                    p = products[pid]; score = predicted_df.loc[u, pid]
                    with rcols[i%3]:
                        st.markdown(f'<div class="product-card"><span class="product-emoji">{p["emoji"]}</span><div class="product-name">{p["name"]}</div><div style="color:#e74c3c;font-weight:700;">₹{p["price"]}</div><div class="product-score">⭐ Score: {score:.2f}</div></div>', unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cf_{pid}_{u}"):
                            add_to_cart(pid, products); st.toast(f"✅ {p['name']} added!", icon="🛒")

    with tab2:
        st.markdown('<div class="section-header">🔗 Item-Item Similarity</div>', unsafe_allow_html=True)
        sel_product = st.selectbox("Select a product", product_ids, format_func=lambda x: f"{products[x]['emoji']} {products[x]['name']}")
        if st.button("🔍 Find Similar Products"):
            base_cat = products[sel_product]["category"]
            allowed  = RELATED_CATEGORIES.get(base_cat, [base_cat])
            sim_pids, sim_scores = get_similar_products(sel_product, item_sim_df, products, n_recs, filter_categories=allowed)
            st.markdown(f'<div class="section-header">Products similar to {products[sel_product]["name"]}</div>', unsafe_allow_html=True)
            if not sim_pids: st.info("No similar products found.")
            scols = st.columns(3)
            for i,(pid,score) in enumerate(zip(sim_pids, sim_scores)):
                if pid not in products: continue
                p = products[pid]
                with scols[i%3]:
                    st.markdown(f'<div class="product-card"><span class="product-emoji">{p["emoji"]}</span><div class="product-name">{p["name"]}</div><div style="color:#e74c3c;font-weight:700;">₹{p["price"]}</div><div class="product-score">🔗 Similarity: {score:.3f}</div></div>', unsafe_allow_html=True)
                    if st.button("🛒 Add to Cart", key=f"sim_{pid}_{sel_product}"):
                        add_to_cart(pid, products); st.toast(f"✅ {p['name']} added!", icon="🛒")


# ══════════════════════════════════════════════════════════════
# PAGE: IMAGE SCANNER
# ══════════════════════════════════════════════════════════════
elif page == "📸 Image Scanner":
    st.markdown('<h1 class="main-title">📸 Product Image Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grocery photo → CV identifies it → Recommends similar products</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("📌 Upload any grocery/food product image. The CV module analyzes it and maps it to products in our catalog.")

    up_col, prev_col = st.columns([1,1])
    with up_col:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a grocery product image", type=["jpg","jpeg","png","webp"])

    if uploaded_file:
        image     = Image.open(uploaded_file).convert("RGB")
        img_bytes = uploaded_file.getvalue()
        with prev_col:
            st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        btn1, btn2 = st.columns([1,1])
        with btn1: analyze = st.button("🔍 Analyze & Recommend", use_container_width=True)
        with btn2:
            if st.button("🔄 Change Image", use_container_width=True):
                st.session_state["cv_done"] = False; st.rerun()

        if analyze:
            pb = st.progress(0, text="🧠 Initializing...")
            time.sleep(0.3)
            pb.progress(25, text="🤖 Vision API analyzing image...")
            tags_raw, err = classify_image_with_hf(img_bytes)
            pb.progress(70, text="🔍 Matching products in catalog...")
            time.sleep(0.2)

            if tags_raw:
                matched_pids = find_products_from_tags(tags_raw, products)
                pb.progress(100, text="✅ Done!"); time.sleep(0.3); pb.empty()
                st.session_state["cv_tags"]   = tags_raw
                st.session_state["cv_pids"]   = matched_pids
                st.session_state["cv_method"] = "✨ Gemini / HF Vision"
                st.session_state["cv_done"]   = True
            else:
                pb.progress(85, text="🎨 Using color-based fallback...")
                time.sleep(0.3)
                fallback_tags = fallback_color_analysis(image)
                matched_pids  = find_products_from_tags(fallback_tags, products)
                pb.progress(100, text="✅ Done!"); time.sleep(0.3); pb.empty()
                st.session_state["cv_tags"]   = fallback_tags
                st.session_state["cv_pids"]   = matched_pids
                st.session_state["cv_method"] = "🎨 Color-Based Fallback"
                st.session_state["cv_done"]   = True
                if err: st.warning(f"⚠️ Vision API: {err}. Color fallback used.")

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
                tn = item.get("tag",""); conf = item.get("confidence",0); ci = int(conf)
                tags_html += f'<div style="margin:4px 0;"><span class="badge badge-orange">{tn}</span><span class="badge badge-conf">{conf:.0f}%</span><div class="conf-bar-wrap"><div class="conf-bar" style="width:{ci}%;"></div></div></div>'
            else:
                tags_html += f'<span class="badge badge-orange">{item}</span>'
        st.markdown(f'<div style="margin:0.75rem 0;">{tags_html}</div>', unsafe_allow_html=True)

        # Debug log
        if st.session_state.get("cv_debug"):
            with st.expander("🔧 Debug Log — API Status"):
                for msg in st.session_state["cv_debug"]:
                    st.markdown(f"`{msg}`")

        explicit_not_in_catalog = any(
            isinstance(t, dict) and normalize_tag(t.get("tag","")) in ("not in catalog", "not_in_catalog", "notincatalog")
            for t in tags
        )
        top_conf = max([float(t.get("confidence", 0) or 0) for t in tags if isinstance(t, dict)], default=0)
        is_not_in_catalog = explicit_not_in_catalog or (len(tags) > 0 and top_conf < 20)

        st.markdown('<div class="section-header">🛒 Matched Products</div>', unsafe_allow_html=True)
        if is_not_in_catalog:
            st.warning("🚫 This item isn't sold in our store — we only carry packaged grocery products (no loose fresh fruits, vegetables, or salads). Try uploading a packaged product instead!")
        elif not matched:
            st.info("No matching products found. Try a different image.")
        else:
            m_cols = st.columns(3)
            for i, pid in enumerate(matched):
                p = products.get(pid)
                if not p: continue
                with m_cols[i%3]:
                    st.markdown(f'<div class="product-card"><span class="product-emoji">{p["emoji"]}</span><div class="product-name">{p["name"]}</div><div style="color:#e74c3c;font-weight:700;">₹{p["price"]}</div><small style="color:#7f8c8d;">{p["category"]}</small></div>', unsafe_allow_html=True)
                    if st.button("🛒 Add", key=f"cv_match_{pid}"):
                        add_to_cart(pid, products); st.toast(f"✅ {p['name']} added!", icon="🛒")

        if matched and not is_not_in_catalog:
            st.markdown('<div class="section-header">🤖 CF-Enhanced Suggestions</div>', unsafe_allow_html=True)
            base_cat = products.get(matched[0], {}).get("category","")
            blocked_words = ["energy","bisleri","sting","limca","7up","appy","red bull","soda","cola"]
            def is_relevant(pid):
                name = products.get(pid,{}).get("name","").lower()
                return not any(b in name for b in blocked_words)

            sim_pids_all, sim_scores_all = get_similar_products(matched[0], item_sim_df, products, n_recs*2, filter_categories=[base_cat])
            sim_pids   = [p for p in sim_pids_all if is_relevant(p)][:n_recs]
            sim_scores = [sim_scores_all[sim_pids_all.index(p)] for p in sim_pids]

            if not sim_pids:
                st.info("No CF suggestions found.")
            else:
                s_cols = st.columns(4)
                for i,(pid,score) in enumerate(zip(sim_pids, sim_scores)):
                    if pid not in products: continue
                    p = products[pid]
                    with s_cols[i%4]:
                        st.markdown(f'<div class="product-card"><span class="product-emoji">{p["emoji"]}</span><div class="product-name">{p["name"]}</div><div style="color:#e74c3c;font-weight:700;">₹{p["price"]}</div><div class="product-score">🔗 {score:.3f}</div></div>', unsafe_allow_html=True)
                        if st.button("🛒 Add to Cart", key=f"cv_cf_{pid}_{i}"):
                            add_to_cart(pid, products); st.toast(f"✅ {p['name']} added!", icon="🛒")


# ══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
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
            cat_counts[p["category"]] = cat_counts.get(p["category"],0) + 1
        st.bar_chart(pd.DataFrame({"Category":list(cat_counts.keys()),"Count":list(cat_counts.values())}).set_index("Category"))

    st.markdown('<div class="section-header">📈 User Purchase Heatmap (Sample)</div>', unsafe_allow_html=True)
    sample = ratings_df.iloc[:15,:10].copy()
    sample.columns = [f"{products[c]['emoji']}{products[c]['name'][:8]}" for c in sample.columns if c in products]
    st.dataframe(sample.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown('<div class="section-header">🔬 SVD Explained Variance</div>', unsafe_allow_html=True)
    n_comp   = min(10, ratings_df.shape[0]-1, ratings_df.shape[1]-1)
    svd_test = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_test.fit(ratings_df.values)
    var_df = pd.DataFrame({
        "Component": [f"C{i+1}" for i in range(n_comp)],
        "Explained Variance (%)": (svd_test.explained_variance_ratio_*100).round(2)
    })
    st.bar_chart(var_df.set_index("Component"))
    st.caption(f"Total variance explained: {svd_test.explained_variance_ratio_.sum()*100:.1f}%")
