# 📦 Dataset Setup

This project uses the **Instacart Market Basket Analysis** dataset from Kaggle.

> ⚠️ The CSV files are **not included** in this repo due to large file size (~1.3 GB total).

---

## 🔽 How to Download

1. Go to Kaggle: [https://www.kaggle.com/c/instacart-market-basket-analysis/data](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
2. Sign in / create a free Kaggle account
3. Click **"Download All"**
4. Extract the zip — you'll get a folder with these files

---

## 📁 Required Files

Place these CSV files in the `/content/` folder (if using Google Colab) or update the paths in the notebook:

| File | Size (approx) | Description |
|---|---|---|
| `orders.csv` | ~106 MB | All orders with user_id, timestamps |
| `order_products__prior.csv` | ~564 MB | Products in each prior order (main training data) |
| `order_products__train.csv` | ~24 MB | Products in training orders (evaluation data) |
| `products.csv` | ~2.1 MB | Product names and department info |
| `departments.csv` | ~1 KB | Department IDs and names |
| `aisles.csv` | ~3 KB | Aisle IDs and names |

---

## 📊 Dataset Stats

| Metric | Value |
|---|---|
| Total Orders | 3,421,083 |
| Total Products | 49,688 |
| Total Users | 206,209 |
| Users used in this project | 4,628 (user_id ≤ 5000) |

---

## 🚀 Quick Setup (Google Colab)

```python
# Option 1: Upload manually to Colab
from google.colab import files
uploaded = files.upload()
# Upload: orders.csv, products.csv, order_products__prior.csv, order_products__train.csv

# Option 2: Use Kaggle API in Colab
!pip install kaggle
!kaggle competitions download -c instacart-market-basket-analysis
!unzip instacart-market-basket-analysis.zip
```

---

## 📂 After Downloading — Folder Structure

```
/content/
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── departments.csv
└── aisles.csv
```
