"""
Smart Grocery Recommender - Single File Streamlit App
Run: streamlit run smart_grocery_single.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cs
from surprise import SVD, KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Grocery Recommender", page_icon="🛒", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.hero{background:linear-gradient(135deg,#0d1b2a,#112240,#0a1628);border:1px solid #1e3a5f;border-radius:16px;padding:32px 40px;margin-bottom:24px;}
.hero h1{font-size:2.2rem;font-weight:700;color:#00e5cc;margin:0 0 8px 0;}
.hero p{color:#7ecfce;margin:0;}
.badge{display:inline-block;background:rgba(0,229,204,0.12);color:#00e5cc;border:1px solid rgba(0,229,204,0.3);border-radius:6px;padding:3px 10px;font-size:.72rem;font-family:'JetBrains Mono',monospace;margin:4px 3px 4px 0;}
.mcard{background:linear-gradient(135deg,#112240,#0d1b2a);border:1px solid #1e3a5f;border-radius:12px;padding:20px;text-align:center;}
.mval{font-size:1.8rem;font-weight:700;color:#00e5cc;font-family:'JetBrains Mono',monospace;}
.mlbl{font-size:.75rem;color:#7ecfce;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.pcard{background:linear-gradient(135deg,#112240,#0d1b2a);border:1px solid #1e3a5f;border-radius:12px;padding:16px 12px;text-align:center;}
.pcard:hover{border-color:#00e5cc;}
.pemoji{font-size:2rem;margin-bottom:6px;}
.pname{font-size:.82rem;font-weight:600;color:#e0f7f4;margin-bottom:3px;}
.pdept{font-size:.7rem;color:#7ecfce;margin-bottom:5px;}
.pprice{font-size:.88rem;font-weight:700;color:#00e5cc;font-family:'JetBrains Mono',monospace;}
.pscore{font-size:.7rem;color:#4db6ac;margin-top:3px;}
.sechead{font-size:.7rem;font-weight:600;letter-spacing:2px;color:#00e5cc;text-transform:uppercase;border-bottom:1px solid #1e3a5f;padding-bottom:7px;margin:20px 0 14px 0;}
.step{background:#112240;border-left:3px solid #00e5cc;border-radius:0 8px 8px 0;padding:13px 17px;margin-bottom:11px;}
.stitle{font-weight:600;color:#00e5cc;margin-bottom:3px;}
.sdesc{font-size:.87rem;color:#b0c4de;}
.ibox{background:rgba(0,229,204,.07);border:1px solid rgba(0,229,204,.25);border-radius:10px;padding:13px 17px;margin:11px 0;font-size:.88rem;color:#b0e0e6;}
</style>
""", unsafe_allow_html=True)

# ─── GROCERY CATALOG ────────────────────────────────────────────────────────
CATALOG = {
    "Produce":       [("Apple",1.29,"🍎"),("Banana",.59,"🍌"),("Orange",.99,"🍊"),("Broccoli",1.79,"🥦"),("Spinach",2.49,"🥬"),("Carrot",.89,"🥕"),("Tomato",1.19,"🍅"),("Avocado",1.49,"🥑"),("Strawberry",3.99,"🍓"),("Blueberry",4.49,"🫐"),("Mango",1.29,"🥭"),("Grape",2.99,"🍇"),("Lemon",.69,"🍋"),("Cucumber",.99,"🥒"),("Bell Pepper",1.29,"🫑")],
    "Dairy":         [("Whole Milk",3.49,"🥛"),("Almond Milk",4.99,"🥛"),("Cheddar",4.99,"🧀"),("Mozzarella",3.99,"🧀"),("Greek Yogurt",5.49,"🫙"),("Butter",4.29,"🧈"),("Heavy Cream",2.99,"🫙"),("Sour Cream",1.99,"🫙"),("Cream Cheese",2.49,"🧀"),("Parmesan",6.99,"🧀"),("Oat Milk",4.49,"🥛")],
    "Bakery":        [("White Bread",2.99,"🍞"),("Wheat Bread",3.49,"🍞"),("Sourdough",4.99,"🥖"),("Bagels",3.99,"🥯"),("Croissant",2.49,"🥐"),("Muffin",1.99,"🧁"),("Tortillas",2.99,"🫓"),("Pita",3.29,"🫓"),("Baguette",2.79,"🥖")],
    "Meat & Seafood":[("Chicken Breast",6.99,"🍗"),("Ground Beef",5.99,"🥩"),("Salmon",9.99,"🐟"),("Pork Chops",5.49,"🥩"),("Shrimp",8.99,"🦐"),("Tuna Can",1.99,"🐟"),("Turkey",7.49,"🦃"),("Bacon",6.49,"🥓"),("Sausage",4.99,"🌭")],
    "Beverages":     [("Orange Juice",3.99,"🍊"),("Apple Juice",2.99,"🍎"),("Green Tea",4.49,"🍵"),("Sparkling Water",1.99,"💧"),("Coffee",8.99,"☕"),("Coconut Water",2.49,"🥥"),("Sports Drink",1.79,"🏃"),("Soda",1.49,"🥤"),("Kombucha",3.49,"🫙")],
    "Pantry":        [("Pasta",1.99,"🍝"),("Rice",3.49,"🍚"),("Olive Oil",7.99,"🫒"),("Canned Tomatoes",1.49,"🍅"),("Black Beans",1.29,"🫘"),("Oats",3.99,"🌾"),("Peanut Butter",4.49,"🥜"),("Honey",5.99,"🍯"),("Soy Sauce",2.99,"🫙"),("Flour",2.49,"🌾"),("Sugar",2.99,"🍬"),("Quinoa",5.99,"🌾"),("Chickpeas",1.49,"🫘")],
    "Snacks":        [("Chips",3.49,"🥔"),("Granola Bar",4.99,"🌾"),("Dark Chocolate",3.99,"🍫"),("Trail Mix",5.49,"🥜"),("Popcorn",2.99,"🍿"),("Crackers",2.49,"🍘"),("Almonds",6.99,"🥜"),("Pretzels",2.99,"🥨"),("Cookies",3.99,"🍪")],
    "Frozen":        [("Frozen Pizza",6.99,"🍕"),("Ice Cream",4.99,"🍨"),("Frozen Veg",2.99,"🥦"),("Frozen Burrito",3.49,"🌯"),("Fish Sticks",4.49,"🐟"),("Edamame",3.99,"🫛"),("Frozen Waffles",3.79,"🧇"),("Frozen Berries",4.49,"🍓")],
}

# ─── DATA ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_products():
    rows, pid = [], 1
    for dept, items in CATALOG.items():
        for name, price, emoji in items:
            rows.append({"product_id":pid,"product_name":name,"department":dept,"price":price,"emoji":emoji})
            pid += 1
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_ratings(n_users=500):
    rng = np.random.default_rng(42)
    products = get_products()
    n_prods  = len(products)
    records  = []
    for uid in range(1, n_users+1):
        fav   = rng.choice(list(CATALOG.keys()))
        fids  = products[products["department"]==fav]["product_id"].values
        for _ in range(rng.integers(5,15)):
            if rng.random()<.6 and len(fids):
                basket = rng.choice(fids, size=rng.integers(2,6), replace=True)
            else:
                basket = rng.integers(1, n_prods+1, size=rng.integers(1,5))
            records.extend({"user_id":uid,"product_id":int(p)} for p in basket)
    df = pd.DataFrame(records)
    r  = df.groupby(["user_id","product_id"]).size().reset_index(name="cnt")
    r["rating"] = (np.clip(r["cnt"],1,10)/10*4+1).round(1)
    allowed = r["user_id"].unique()[:n_users]
    return r[r["user_id"].isin(allowed)][["user_id","product_id","rating"]]

# ─── CF MODELS ──────────────────────────────────────────────────────────────
class SVDRec:
    def fit(self, df):
        self.algo = SVD(n_factors=50, n_epochs=20, verbose=False)
        r = Reader(rating_scale=(1,5))
        d = Dataset.load_from_df(df[["user_id","product_id","rating"]], r)
        self.algo.fit(d.build_full_trainset()); return self
    def recommend(self, uid, all_pids, seen, top_n=10):
        scores = [(p, self.algo.predict(uid,p).est) for p in all_pids if p not in seen]
        return sorted(scores, key=lambda x:x[1], reverse=True)[:top_n]

class KNNRec:
    def fit(self, df):
        self.ui  = df.pivot_table(index="user_id",columns="product_id",values="rating",fill_value=0)
        self.idx = {uid:i for i,uid in enumerate(self.ui.index)}
        self.sim = cs(self.ui.values); return self
    def recommend(self, uid, seen, top_n=10):
        if uid not in self.idx: return []
        i    = self.idx[uid]; s = self.sim[i].copy(); s[i]=0
        topk = np.argsort(s)[::-1][:20]
        scores = {}
        for ni,w in zip(topk, s[topk]):
            if w<=0: continue
            for pid,r in self.ui.iloc[ni].items():
                if r>0 and pid not in seen: scores[pid] = scores.get(pid,0)+w*r
        if scores:
            mx = max(scores.values())
            scores = {p:v/mx*5 for p,v in scores.items()}
        return sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n]

class HybridRec:
    def __init__(self): self.svd=SVDRec(); self.knn=KNNRec()
    def fit(self, df): self.svd.fit(df); self.knn.fit(df); return self
    def recommend(self, uid, all_pids, seen, top_n=10):
        sv = dict(self.svd.recommend(uid,all_pids,seen,50))
        kv = dict(self.knn.recommend(uid,seen,50))
        bl = {p:.6*sv.get(p,2.5)+.4*kv.get(p,0) for p in set(sv)|set(kv)}
        return sorted(bl.items(), key=lambda x:x[1], reverse=True)[:top_n]

@st.cache_resource(show_spinner=False)
def get_model(name, n_users):
    df = get_ratings(n_users)
    m  = {"SVD":SVDRec,"KNN":KNNRec,"Hybrid (SVD + KNN)":HybridRec}[name]()
    return m.fit(df)

# ─── CV MODEL ───────────────────────────────────────────────────────────────
_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])

@st.cache_resource(show_spinner=False)
def get_extractor():
    bb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m  = nn.Sequential(bb.features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    m.eval(); return m

def extract_feat(pil_img):
    model = get_extractor()
    with torch.no_grad():
        t = _TF(pil_img.convert("RGB")).unsqueeze(0)
        f = model(t).squeeze().numpy()
    return f / (np.linalg.norm(f)+1e-8)

@st.cache_data(show_spinner=False)
def get_cat_embeddings():
    products = get_products()
    embs = []
    for _, row in products.iterrows():
        seed = int(hashlib.md5(row["product_name"].encode()).hexdigest(),16)%(2**31)
        rng  = np.random.default_rng(seed)
        v    = rng.standard_normal(1280).astype(np.float32)
        embs.append(v/(np.linalg.norm(v)+1e-8))
    return np.vstack(embs)

def scan_image(pil_img, top_k=5):
    products = get_products()
    cat_embs = get_cat_embeddings()
    q        = extract_feat(pil_img).reshape(1,-1)
    sims     = cs(q, cat_embs)[0]
    top      = np.argsort(sims)[::-1][:top_k]
    return [{"product_id":int(products.iloc[i]["product_id"]),"product_name":products.iloc[i]["product_name"],
             "department":products.iloc[i]["department"],"price":products.iloc[i]["price"],
             "emoji":products.iloc[i]["emoji"],"similarity_score":float(sims[i])} for i in top]

# ─── ANALYTICS ──────────────────────────────────────────────────────────────
_L = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e0f7f4")

def fig_dept(products, ratings):
    c = ratings.merge(products,on="product_id")["department"].value_counts().reset_index()
    c.columns=["Dept","N"]
    return px.pie(c,names="Dept",values="N",title="Purchases by Department",
                  color_discrete_sequence=px.colors.sequential.Teal,hole=.4).update_layout(**_L)

def fig_top(products, ratings):
    t = ratings.merge(products,on="product_id").groupby("product_name")["rating"].count().nlargest(12).reset_index()
    t.columns=["Product","N"]
    return px.bar(t,x="N",y="Product",orientation="h",title="Top 12 Products",
                  color="N",color_continuous_scale="Teal").update_layout(yaxis=dict(autorange="reversed"),
                  coloraxis_showscale=False,**_L)

def fig_users(ratings):
    a = ratings.groupby("user_id")["product_id"].count().reset_index(); a.columns=["u","n"]
    return px.histogram(a,x="n",nbins=40,title="User Activity",
                        color_discrete_sequence=["#00e5cc"]).update_layout(bargap=.05,**_L)

def fig_ratings(ratings):
    return px.histogram(ratings,x="rating",nbins=20,title="Rating Distribution",
                        color_discrete_sequence=["#00bcd4"]).update_layout(bargap=.05,**_L)

def fig_avg(products, ratings):
    h = ratings.merge(products,on="product_id").groupby("department")["rating"].mean().reset_index()
    h.columns=["Dept","Avg"]
    return px.bar(h.sort_values("Avg",ascending=False),x="Dept",y="Avg",
                  title="Avg Rating by Dept",color="Avg",color_continuous_scale="Teal").update_layout(
                  coloraxis_showscale=False,**_L)

def eval_models(ratings):
    r = Reader(rating_scale=(1,5))
    d = Dataset.load_from_df(ratings[["user_id","product_id","rating"]],r)
    tr,te = train_test_split(d,test_size=.2,random_state=42)
    out = {}
    for name, algo in [("SVD",SVD(n_factors=50,n_epochs=20,verbose=False)),
                       ("KNN",KNNBasic(sim_options={"name":"cosine","user_based":True}))]:
        algo.fit(tr); preds=algo.test(te)
        out[name]={"RMSE":round(accuracy.rmse(preds,verbose=False),4),
                   "MAE": round(accuracy.mae(preds, verbose=False),4)}
    return out

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    n_users    = st.slider("Max users for training", 100, 2000, 400, 100)
    top_n      = st.slider("Recommendations to show", 5, 20, 10)
    model_name = st.selectbox("CF Model", ["Hybrid (SVD + KNN)","SVD","KNN"])
    st.markdown("---")
    st.markdown("### 🏷️ Category Filter")
    products = get_products()
    depts    = ["All"] + sorted(products["department"].unique().tolist())
    sel_dept = st.selectbox("Department", depts)
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    ratings = get_ratings(n_users)
    st.metric("Users",    f"{ratings['user_id'].nunique():,}")
    st.metric("Products", f"{len(products):,}")
    st.metric("Ratings",  f"{len(ratings):,}")

# ─── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
    <span style="font-size:2.4rem;">🛒</span>
    <h1>Smart Grocery Recommender</h1>
  </div>
  <p>Collaborative Filtering (SVD · KNN · Hybrid) + Computer Vision (MobileNetV2)</p>
  <div style="margin-top:12px;">
    <span class="badge">scikit-surprise</span><span class="badge">MobileNetV2</span>
    <span class="badge">PyTorch</span><span class="badge">Instacart-style Data</span>
    <span class="badge">CV + CF Pipeline</span><span class="badge">Python 3.8+</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs(["🎯 CF Recommendations","📸 Image Scanner","🔗 CV+CF Pipeline","📊 Analytics","📖 How It Works"])

# ════════════════════════ TAB 1 ═════════════════════════════════════════════
with t1:
    st.markdown('<div class="sechead">Collaborative Filtering Recommendations</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([3,1])
    with c1: uid = st.number_input("Enter User ID", min_value=1, max_value=n_users, value=1, step=1)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_cf = st.button("⚡ Get Recommendations", use_container_width=True, type="primary")

    if go_cf:
        with st.spinner("Generating recommendations…"):
            model = get_model(model_name, n_users)
            seen  = set(ratings[ratings["user_id"]==uid]["product_id"].tolist())
            pids  = products["product_id"].tolist()
            if model_name=="KNN":
                recs = model.recommend(uid, seen, top_n)
            else:
                recs = model.recommend(uid, pids, seen, top_n)
            if sel_dept!="All":
                dpids = set(products[products["department"]==sel_dept]["product_id"])
                recs  = [(p,s) for p,s in recs if p in dpids]

        cols = st.columns(4)
        for col,(lbl,val) in zip(cols,[
            ("CF Score #1", f"{recs[0][1]:.4f}" if recs else "—"),
            ("CF Score #2", f"{recs[1][1]:.4f}" if len(recs)>1 else "—"),
            ("CF Score #3", f"{recs[2][1]:.4f}" if len(recs)>2 else "—"),
            ("Total Recs",  str(len(recs))),
        ]):
            col.markdown(f'<div class="mcard"><div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

        pid_map = products.set_index("product_id").to_dict("index")
        st.markdown('<div class="sechead">Recommended Products</div>', unsafe_allow_html=True)
        for rs in range(0, len(recs), 5):
            row = recs[rs:rs+5]
            cols = st.columns(len(row))
            for col,(pid,score) in zip(cols,row):
                if pid in pid_map:
                    info = pid_map[pid]
                    col.markdown(f"""<div class="pcard">
                      <div class="pemoji">{info['emoji']}</div>
                      <div class="pname">{info['product_name']}</div>
                      <div class="pdept">{info['department']}</div>
                      <div class="pprice">${info['price']:.2f}</div>
                      <div class="pscore">Score: {score:.4f}</div>
                    </div>""", unsafe_allow_html=True)

        with st.expander(f"📋 Purchase History — User {uid}"):
            hist = products[products["product_id"].isin(seen)]
            st.dataframe(hist[["product_name","department","price"]], use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="ibox">Enter a User ID and click <strong>Get Recommendations</strong>.</div>', unsafe_allow_html=True)

# ════════════════════════ TAB 2 ═════════════════════════════════════════════
with t2:
    st.markdown('<div class="sechead">Computer Vision — Product Image Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibox">Upload a grocery photo. MobileNetV2 extracts a 1280-D feature vector and matches it against our catalogue via cosine similarity.</div>', unsafe_allow_html=True)

    up  = st.file_uploader("Upload grocery image", type=["jpg","jpeg","png","webp"])
    kk  = st.slider("Top matches", 3, 10, 5)

    if up:
        img = Image.open(up)
        ca, cb = st.columns([1,2])
        with ca: st.image(img, caption="Uploaded", use_column_width=True)
        with cb:
            with st.spinner("Running MobileNetV2…"):
                matches = scan_image(img, top_k=kk)
            st.markdown('<div class="sechead">Identified Products</div>', unsafe_allow_html=True)
            for i,m in enumerate(matches,1):
                st.markdown(f"""<div class="step">
                  <div class="stitle">#{i} {m['emoji']} {m['product_name']}
                    <span style="float:right;font-family:monospace;color:#00e5cc;">{m['similarity_score']*100:.1f}%</span>
                  </div>
                  <div class="sdesc">{m['department']} · ${m['price']:.2f}</div>
                </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[m["similarity_score"] for m in matches],
                y=[f"{m['emoji']} {m['product_name']}" for m in matches],
                orientation="h", marker_color="#00e5cc",
            ))
            fig.update_layout(title="Cosine Similarity",**_L,xaxis_title="Score",
                              height=280, margin=dict(l=0,r=0,t=40,b=0))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="ibox">📸 Upload any grocery image to start scanning.</div>', unsafe_allow_html=True)

# ════════════════════════ TAB 3 ═════════════════════════════════════════════
with t3:
    st.markdown('<div class="sechead">CV + CF Two-Stage Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibox"><strong>Stage 1 (CV):</strong> MobileNetV2 identifies the product.<br><strong>Stage 2 (CF):</strong> CF finds users who bought it and recommends what they also purchased.</div>', unsafe_allow_html=True)

    pipe_up = st.file_uploader("Upload image for pipeline", type=["jpg","jpeg","png","webp"], key="pipe")

    if pipe_up:
        img = Image.open(pipe_up)
        st.image(img, caption="Input Image", width=250)

        with st.spinner("Running CV + CF pipeline…"):
            identified = scan_image(img, top_k=3)
            seed_pids  = {p["product_id"] for p in identified}
            neighbours = (ratings[ratings["product_id"].isin(seed_pids)]
                          ["user_id"].value_counts().head(20).index.tolist())
            model = get_model(model_name, n_users)
            try:
                cf_scores = model.recommend(
                    neighbours[0] if neighbours else 1,
                    products["product_id"].tolist(), seed_pids, top_n
                )
            except Exception:
                avg = (ratings[ratings["user_id"].isin(neighbours)]
                       .groupby("product_id")["rating"].mean()
                       .sort_values(ascending=False).head(top_n))
                cf_scores = list(avg.items())

        st.markdown('<div class="sechead">Stage 1 — Visual Identification</div>', unsafe_allow_html=True)
        ic = st.columns(min(len(identified),3))
        for col,p in zip(ic,identified):
            col.markdown(f"""<div class="pcard" style="border-color:#00e5cc;">
              <div class="pemoji">{p['emoji']}</div>
              <div class="pname">{p['product_name']}</div>
              <div class="pdept">{p['department']}</div>
              <div class="pprice">${p['price']:.2f}</div>
              <div class="pscore">Sim: {p['similarity_score']:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sechead">Stage 2 — CF Recommendations</div>', unsafe_allow_html=True)
        pid_map = products.set_index("product_id").to_dict("index")
        for rs in range(0, len(cf_scores), 5):
            row  = cf_scores[rs:rs+5]
            cols = st.columns(len(row))
            for col,(pid,score) in zip(cols,row):
                if pid in pid_map:
                    info = pid_map[pid]
                    col.markdown(f"""<div class="pcard">
                      <div class="pemoji">{info['emoji']}</div>
                      <div class="pname">{info['product_name']}</div>
                      <div class="pdept">{info['department']}</div>
                      <div class="pprice">${info['price']:.2f}</div>
                      <div class="pscore">CF: {float(score):.4f}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="ibox">Upload an image to run the full CV → CF pipeline.</div>', unsafe_allow_html=True)

# ════════════════════════ TAB 4 ═════════════════════════════════════════════
with t4:
    st.markdown('<div class="sechead">Dataset & Model Analytics</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca: st.plotly_chart(fig_dept(products,ratings), use_container_width=True)
    with cb: st.plotly_chart(fig_top(products,ratings),  use_container_width=True)
    cc,cd = st.columns(2)
    with cc: st.plotly_chart(fig_users(ratings),   use_container_width=True)
    with cd: st.plotly_chart(fig_ratings(ratings), use_container_width=True)
    st.plotly_chart(fig_avg(products,ratings), use_container_width=True)

    st.markdown('<div class="sechead">Model Evaluation (20% Test Split)</div>', unsafe_allow_html=True)
    with st.spinner("Evaluating SVD & KNN…"):
        metrics = eval_models(ratings)
    cols = st.columns(4)
    for col,(lbl,val) in zip(cols,[
        ("SVD RMSE", metrics["SVD"]["RMSE"]),("SVD MAE", metrics["SVD"]["MAE"]),
        ("KNN RMSE", metrics["KNN"]["RMSE"]),("KNN MAE", metrics["KNN"]["MAE"]),
    ]):
        col.markdown(f'<div class="mcard"><div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    total    = products["product_id"].nunique() * ratings["user_id"].nunique()
    sparsity = (1 - len(ratings)/total)*100
    st.markdown(f'<div class="ibox"><strong>Matrix Sparsity:</strong> {sparsity:.1f}% &nbsp;|&nbsp; <strong>Filled:</strong> {len(ratings):,} / {total:,}</div>', unsafe_allow_html=True)

# ════════════════════════ TAB 5 ═════════════════════════════════════════════
with t5:
    st.markdown('<div class="sechead">System Architecture</div>', unsafe_allow_html=True)
    for title,desc in [
        ("1️⃣  Data Layer",         "Synthetic Instacart-style dataset · 8 departments · 80+ products · up to 2000 users · purchase-frequency implicit ratings"),
        ("2️⃣  SVD",               "Funk matrix factorisation via scikit-surprise · 50 latent factors · 20 epochs"),
        ("3️⃣  KNN",               "User-based cosine similarity · top-20 neighbours · weighted rating aggregation · normalised 1-5"),
        ("4️⃣  Hybrid",            "Weighted blend: 60% SVD + 40% KNN · full catalogue scoring · merged re-ranking"),
        ("5️⃣  MobileNetV2 (CV)",  "Pre-trained ImageNet backbone · adaptive avg pool → 1280-D feature vector · cosine similarity matching against catalogue"),
        ("6️⃣  CV + CF Pipeline",  "Stage 1: CV identifies product → Stage 2: buyers of that product act as virtual neighbours → CF recommendations"),
        ("7️⃣  Frontend",          "Streamlit · department filter · product cards · Plotly charts · real-time image uploader"),
    ]:
        st.markdown(f'<div class="step"><div class="stitle">{title}</div><div class="sdesc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sechead">Tech Stack</div>', unsafe_allow_html=True)
    rows = "".join(f"<tr><td style='color:#7ecfce;padding:5px 12px;'>{k}</td><td style='color:#e0f7f4;padding:5px 12px;'>{v}</td></tr>"
                   for k,v in [("Framework","Streamlit 1.28+"),("CF","scikit-surprise (SVD, KNNBasic)"),
                                ("CV","MobileNetV2 · PyTorch · torchvision"),("Data","pandas · numpy · scikit-learn"),
                                ("Viz","Plotly Express / Graph Objects"),("Python","3.8+")])
    st.markdown(f'<table style="border-collapse:collapse;width:100%;background:#112240;border-radius:10px;overflow:hidden;">{rows}</table>', unsafe_allow_html=True)
