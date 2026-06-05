📦 Dataset Setup
This project uses the Instacart Market Basket Analysis dataset from Kaggle.

⚠️ The CSV files are not included in this repo due to large file size (~1.3 GB total).


🔽 How to Download

Go to Kaggle: https://www.kaggle.com/c/instacart-market-basket-analysis/data
Sign in / create a free Kaggle account
Click "Download All"
Extract the zip — you'll get a folder with these files


📁 Required Files
Place these CSV files in the /content/ folder (if using Google Colab) or update the paths in the notebook:
FileSize (approx)Descriptionorders.csv~106 MBAll orders with user_id, timestampsorder_products__prior.csv~564 MBProducts in each prior order (main training data)order_products__train.csv~24 MBProducts in training orders (evaluation data)products.csv~2.1 MBProduct names and department infodepartments.csv~1 KBDepartment IDs and namesaisles.csv~3 KBAisle IDs and names

📊 Dataset Stats
MetricValueTotal Orders3,421,083Total Products49,688Total Users206,209Users used in this project4,628 (user_id ≤ 5000)

🚀 Quick Setup (Google Colab)
python# Option 1: Upload manually to Colab
from google.colab import files
uploaded = files.upload()
# Upload: orders.csv, products.csv, order_products__prior.csv, order_products__train.csv

# Option 2: Use Kaggle API in Colab
!pip install kaggle
!kaggle competitions download -c instacart-market-basket-analysis
!unzip instacart-market-basket-analysis.zip

📂 After Downloading — Folder Structure
/content/
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── departments.csv
└── aisles.csv
