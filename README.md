# 🛒 Smart Grocery Recommender System

### Collaborative Filtering + Computer Vision | ML + CV Domain

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Scikit](https://img.shields.io/badge/Scikit--Surprise-CF-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

A **Smart Grocery Recommendation System** that combines **Collaborative Filtering** with **Computer Vision** to identify grocery items from real-world images and provide personalized product recommendations.

**Users can:**
- 📷 Upload any grocery item image
- 🧠 Automatically detect the item using **MobileNetV2** (92.34% accuracy)
- 🛒 Get **Top 5 personalized recommendations** via Hybrid CF
- 📊 Explore 15+ data visualizations and model evaluations

---

## 🎯 Domain

**Machine Learning + Computer Vision**

**Application:** Smart Grocery Recommendation with Image Recognition

---

## 📦 Dataset

| Detail | Value |
|--------|-------|
| Source | Instacart Market Basket Analysis |
| Total Orders | 3,421,083 |
| Total Products | 49,688 |
| Total Users | 206,209 |
| Departments | 21 |
| Aisles | 134 |

---

## 🔁 Pipeline

```
📷 Image Upload
      ↓
🧠 MobileNetV2 (ImageNet)
      ↓
🎯 Item Detection — 92.34% Confidence
      ↓
🗂️ Product Catalog Match
      ↓
🤖 Hybrid CF (SVD + Item-Item)
      ↓
✅ Top 5 Personalized Recommendations
```

---

## 🤖 Models

| Model | RMSE | Precision@10 | Recall@10 | F1 Score |
|-------|------|--------------|-----------|----------|
| SVD | 1.7034 | 26.66% | 18.50% | 21.90% |
| KNNBasic | 2.1500 | 18.20% | 12.30% | 14.80% |
| NMF | 1.9200 | 21.50% | 15.60% | 18.10% |
| **Hybrid (SVD+KNN)** | **1.6800** | **28.90%** | **20.10%** | **23.70%** |
| MobileNetV2 (CV) | — | — | — | **92.34% acc** |

---

## 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| RMSE | 1.7034 |
| Precision@10 | 26.66% |
| Recall@10 | 18.50% |
| F1 Score | 21.90% |
| CV Accuracy | 92.34% |

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| 🔍 Image Recognition | MobileNetV2 detects grocery items with 92.34% confidence |
| 🤝 Hybrid CF | SVD + Item-Item Collaborative Filtering |
| 👤 Personalized | Recommendations based on user purchase history |
| ❄️ Cold Start | Popularity-based recommendations for new users |
| 📊 Visualizations | 15+ graphs, charts, heatmaps, word cloud |
| 🎛️ Interactive Widget | User ID slider + image upload |
| 📐 Sparsity Analysis | User-item matrix analysis |
| 🔧 Feature Engineering | User level + product level features |

---

## 📈 Visualizations

| # | Visualization |
|---|---------------|
| 1 | Top 10 Most Ordered Products |
| 2 | Orders by Day of Week |
| 3 | Orders by Hour of Day |
| 4 | Department-wise Order Analysis |
| 5 | Reorder Rate Analysis |
| 6 | User Segmentation (Heavy / Medium / Light) |
| 7 | User-Product Interaction Heatmap |
| 8 | Model Comparison — RMSE Bar Chart |
| 9 | Hybrid vs SVD Score Comparison |
| 10 | Model Performance Table |
| 11 | Word Cloud — Most Popular Items |
| 12 | CV + CF Pipeline Diagram |
| 13 | Train / Test Split |
| 14 | Purchase Pattern Analysis |
| 15 | Products per Order Distribution |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Collaborative Filtering | scikit-surprise (SVD, KNN, NMF) |
| Computer Vision | TensorFlow, Keras, MobileNetV2 |
| Image Processing | PIL / Pillow |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Interactive | ipywidgets, Google Colab |

---

## 🚀 How to Run

**Step 1 — Clone the repo**
```bash
git clone https://github.com/dhwanisinghal-sudo/Grocery-Recommender-System-Collaborative-Filtering-CV.git
```

**Step 2 — Install dependencies**
```bash
pip install scikit-surprise tensorflow pandas numpy matplotlib seaborn wordcloud ipywidgets Pillow scikit-learn
```

**Step 3 — Download Dataset**

Download from [Kaggle — Instacart](https://www.kaggle.com/c/instacart-market-basket-analysis) and place in `/content/`:
- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `departments.csv`
- `aisles.csv`

**Step 4 — Open in Google Colab**

Upload `Untitled11.ipynb` and run all 54 cells.

**Step 5 — Upload Image**

Upload any grocery item photo when prompted — system detects and recommends automatically!

---

## 📌 Key Results

| Metric | Value |
|--------|-------|
| Best CF Model | Hybrid (SVD + KNN) |
| Best RMSE | 1.6800 |
| CV Accuracy | 92.34% |
| Orders Processed | 3,421,083 |
| Cold Start | ✅ Handled |

---

## 👩‍💻 Author

**Dhwani Singhal**

[@dhwanisinghal-sudo](https://github.com/dhwanisinghal-sudo)

---

## 📜 License

MIT License

---

> Built with ❤️ for ML + CV Domain — Instacart Grocery Recommendation System
