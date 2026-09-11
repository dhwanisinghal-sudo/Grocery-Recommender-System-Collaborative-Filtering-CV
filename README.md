# 🛒 Smart Grocery Recommender System
### Collaborative Filtering + Computer Vision | ML + CV Domain

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://camo.githubusercontent.com/46eb2ce89247d240b2828be1b397cffe563c29478bdf3b2fac80d8b0f6a96645/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e382b2d626c75653f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://camo.githubusercontent.com/94b86c1a2f95dac95a95769ac9bb7ceada4bd7b9283019c8a82d39589c31318e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f54656e736f72466c6f772d322e782d6f72616e67653f7374796c653d666f722d7468652d6261646765266c6f676f3d74656e736f72666c6f77)
[![Scikit](https://img.shields.io/badge/Scikit--Surprise-CF-green?style=for-the-badge)](https://camo.githubusercontent.com/e4200b8fa03f050ba7e29c38483c6a18f2f49289342b2e04eac0b683a4270508/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5363696b69742d2d53757270726973652d43462d677265656e3f7374796c653d666f722d7468652d6261646765)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)](https://camo.githubusercontent.com/9b7a2f0cc11cffecf22e34f4cee858f21d3ba3a3111335c21a0bc0af21e8f3dd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374617475732d436f6d706c6574652d627269676874677265656e3f7374796c653d666f722d7468652d6261646765)

### 🚀 [Live App: Try it now!](https://grocery-recommender-system-collaborative-filtering-cv-ln5suvwv.streamlit.app)

---

## 📌 Project Overview

A **Smart Grocery Recommendation System** that combines **Collaborative Filtering** with **Computer Vision** to identify grocery items from real-world images and provide personalized product recommendations.

**🔗 Live App:** [https://grocery-recommender-system-collaborative-filtering-cv-ln5suvwv.streamlit.app](https://grocery-recommender-system-collaborative-filtering-cv-ln5suvwv.streamlit.app)

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

| Detail         | Value                            |
| -------------- | -------------------------------- |
| Source         | Instacart Market Basket Analysis |
| Total Orders   | 3,421,083                        |
| Total Products | 49,688                           |
| Total Users    | 206,209                          |
| Departments    | 21                               |
| Aisles         | 134                              |

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

| Model                | RMSE       | Precision@10 | Recall@10  | F1 Score       |
| --------------------- | ---------- | ------------ | ---------- | -------------- |
| SVD                   | 1.7034     | 26.66%       | 18.50%     | 21.90%         |
| KNNBasic              | 2.1500     | 18.20%       | 12.30%     | 14.80%         |
| NMF                   | 1.9200     | 21.50%       | 15.60%     | 18.10%         |
| **Hybrid (SVD+KNN)**  | **1.6800** | **28.90%**   | **20.10%** | **23.70%**     |
| MobileNetV2 (CV)      | —          | —            | —          | **92.34% acc** |

---

## 📊 Evaluation Metrics

| Metric       | Value  |
| ------------ | ------ |
| RMSE         | 1.7034 |
| Precision@10 | 26.66% |
| Recall@10    | 18.50% |
| F1 Score     | 21.90% |
| CV Accuracy  | 92.34% |

---

## ✅ Features

| Feature                | Description                                              |
| ----------------------- | ---------------------------------------------------------- |
| 🔍 Image Recognition    | MobileNetV2 detects grocery items with 92.34% confidence |
| 🤝 Hybrid CF            | SVD + Item-Item Collaborative Filtering                  |
| 👤 Personalized         | Recommendations based on user purchase history           |
| ❄️ Cold Start          | Popularity-based recommendations for new users           |
| 📊 Visualizations       | 15+ graphs, charts, heatmaps, word cloud                 |
| 🎛️ Interactive Widget  | User ID slider + image upload                             |
| 📐 Sparsity Analysis    | User-item matrix analysis                                |
| 🔧 Feature Engineering  | User level + product level features                       |

---

## 📈 Visualizations

| #   | Visualization                              |
| --- | ------------------------------------------- |
| 1   | Top 10 Most Ordered Products               |
| 2   | Orders by Day of Week                      |
| 3   | Orders by Hour of Day                      |
| 4   | Department-wise Order Analysis             |
| 5   | Reorder Rate Analysis                      |
| 6   | User Segmentation (Heavy / Medium / Light) |
| 7   | User-Product Interaction Heatmap           |
| 8   | Model Comparison — RMSE Bar Chart          |
| 9   | Hybrid vs SVD Score Comparison             |
| 10  | Model Performance Table                    |
| 11  | Word Cloud — Most Popular Items            |
| 12  | CV + CF Pipeline Diagram                   |
| 13  | Train / Test Split                         |
| 14  | Purchase Pattern Analysis                  |
| 15  | Products per Order Distribution            |

---

## 🛠️ Tech Stack

| Category                 | Libraries                       |
| ------------------------- | -------------------------------- |
| Collaborative Filtering  | scikit-surprise (SVD, KNN, NMF) |
| Computer Vision          | TensorFlow, Keras, MobileNetV2  |
| Image Processing         | PIL / Pillow                    |
| Data Processing          | Pandas, NumPy                   |
| Visualization            | Matplotlib, Seaborn, WordCloud  |
| Interactive              | ipywidgets, Google Colab        |
| Deployment               | Streamlit                       |

---

## 📌 Key Results

| Metric            | Value              |
| ------------------ | -------------------- |
| Best CF Model      | Hybrid (SVD + KNN) |
| Best RMSE          | 1.6800             |
| CV Accuracy        | 92.34%              |
| Orders Processed   | 3,421,083           |
| Cold Start         | ✅ Handled          |

---

## 👩‍💻 Author

**Dhwani Singhal**

[@dhwanisinghal-sudo](https://github.com/dhwanisinghal-sudo)

---

> Built with ❤️ for ML + CV Domain — Instacart Grocery Recommendation System
