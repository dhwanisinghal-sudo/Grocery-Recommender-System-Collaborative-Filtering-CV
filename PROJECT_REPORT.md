# 📋 Smart Grocery Recommender System — Project Report

**Domain:** Machine Learning + Computer Vision  
**Dataset:** Instacart Market Basket Analysis  
**Status:** ✅ Complete

---

## 1. Problem Statement

Traditional grocery apps recommend the same popular products to everyone — ignoring individual purchase history. This project builds a **personalized Smart Grocery Recommendation System** that:

- Identifies grocery items from real-world images using **Computer Vision**
- Recommends personalized products using **Collaborative Filtering**
- Combines both into an end-to-end **CV + CF pipeline**

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) |
| Total Orders | 3,421,083 |
| Total Products | 49,688 |
| Total Users | 206,209 |
| Users Used | 4,628 (user_id ≤ 5000) |
| Key Files | `orders.csv`, `order_products__prior.csv`, `products.csv` |

---

## 3. System Architecture

```
📷 Image Upload
      ↓
🧠 MobileNetV2 (CV Model)
      ↓ (92.34% accuracy)
🔍 Catalog Matching (product name → Instacart catalog)
      ↓
🤖 Hybrid Collaborative Filtering (SVD + Item-Item KNN)
      ↓
🛒 Top 5 Personalized Recommendations
```

---

## 4. Models Used

### 4.1 Collaborative Filtering Models

| Model | RMSE | Notes |
|---|---|---|
| SVD | 1.7034 | Best single model |
| KNNBasic | 2.1500 | User-user similarity |
| NMF | 1.9200 | Matrix factorization |
| **Hybrid (SVD + KNN)** | **1.6800** | **Best Overall** |

### 4.2 Computer Vision Model

| Model | Accuracy | Use |
|---|---|---|
| MobileNetV2 (ImageNet) | **92.34%** | Grocery image classification |

- Pre-trained on ImageNet (1000 classes)
- Fine-tuned/used with transfer learning for grocery detection
- Input: any grocery item image
- Output: detected class label → mapped to Instacart catalog

---

## 5. Evaluation Metrics

| Metric | Value |
|---|---|
| RMSE (Hybrid CF) | 1.6800 |
| Precision@10 | 26.66% |
| Recall@10 | 18.50% |
| F1 Score | 21.90% |
| CV Accuracy (MobileNetV2) | 92.34% |

---

## 6. Key Features & Visualizations

The notebook includes **19+ visualizations**:

- Top 10 Most Ordered Products
- Orders by Day of Week & Hour of Day
- Department-wise Product Analysis
- Reorder Rate Distribution
- User Segmentation (Heavy / Medium / Light buyers)
- User-Item Interaction Heatmap
- Model RMSE Comparison Chart
- Hybrid Score Distribution
- Cold Start Problem Solution
- Purchase Pattern Analysis
- Word Cloud of Product Names
- CV + CF Pipeline Diagram
- Train/Test Split Visualization
- Matrix Sparsity Analysis
- Real CV Classification Results

---

## 7. Interactive Demo

The notebook includes an **interactive widget** (ipywidgets):
- Adjust **User ID** via slider (1–5000)
- Select model: `SVD Only` or `Hybrid (SVD + Item-Item)`
- Click **"Get Recommendations"** → instant Top 5 results

---

## 8. Results Summary

- **Hybrid CF** outperforms all individual models (RMSE: 1.68)
- **MobileNetV2** achieves 92.34% accuracy on real grocery images
- **End-to-end pipeline** successfully detects item from image and returns Top 5 recommendations
- Example: Upload banana image → detects "Banana" → recommends: Strawberry Yogurt, Lactose Free Milk, Banana, Greek Yogurt, Almond Milk
