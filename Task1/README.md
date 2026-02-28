# 🛍️ Zara Sales — Exploratory Data Analysis (EDA)

> A comprehensive data analysis project exploring Zara product sales data to uncover pricing patterns, promotion effectiveness, revenue drivers, and key business insights.

---

## 📌 Project Overview

This project performs a full Exploratory Data Analysis (EDA) on the **Zara Sales dataset** sourced from Kaggle. The objective is to thoroughly explore, understand, and analyze the dataset in order to:

- Extract meaningful business insights
- Identify patterns and trends in pricing and sales
- Detect anomalies and outliers
- Understand relationships between product attributes and sales performance

The analysis concludes with a structured business report suitable for both technical and non-technical stakeholders.

---

## 📂 Repository Structure

```
zara-sales-eda/
│
├── Zara_Sales_EDA.ipynb        # Main Jupyter Notebook (full analysis)
├── Zara_sales_EDA.csv          # Dataset (download from Kaggle — link below)
├── zara_sales_eda_report.html  # Auto-generated ydata-profiling report
└── README.md                   # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Zara Sales for EDA](https://www.kaggle.com/datasets/marixe/zara-sales-for-eda) |
| **Rows** | 20,252 products |
| **Columns** | 17 features |
| **Separator** | Semicolon (`;`) |

### Key Columns

| Column | Description |
|---|---|
| `name` | Product name |
| `price` | Product price |
| `Sales Volume` | Units sold |
| `Promotion` | Whether the product is on promotion |
| `Seasonal` | Whether it's a seasonal product |
| `Product Category` | e.g. tops, outerwear, shoes |
| `section` | Male / Female |
| `season` | Which season it belongs to |
| `material` | Fabric/material composition |
| `origin` | Country of origin |
| `terms` | Type of clothing |
| `Product Position` | Position in catalog/store layout |

---

## 🔍 Analysis Outline

The notebook is structured into **8 sections**:

1. **Environment Setup & Data Loading** — Import libraries, load and back up the dataset
2. **Dataset Structure & Overview** — Shape, dtypes, `.info()`, `.describe()`
3. **Data Quality Assessment & Cleaning** — Missing values, duplicates, type casting
4. **Univariate Analysis** — Histograms, boxplots, categorical bar charts
5. **Bivariate & Multivariate Analysis** — Scatter plots, grouped boxplots, correlation heatmap
6. **Outlier Detection** — IQR method + Z-score flagging
7. **Additional Visualizations** — Revenue estimation, pairplot, position analysis, automated EDA via `ydata-profiling`
8. **Key Business Insights & Final Report** — Summary statistics + written business recommendations

---

## 📈 Key Findings

- 📉 **Price is right-skewed** — most products are affordable with a premium tail
- 🏷️ **Promotions drive volume but reduce unit price** — a clear margin vs. volume trade-off
- 🏆 **Top 10 products** drive a disproportionate share of total estimated revenue (Pareto effect)
- 🧵 **Cotton and wool** are the most common materials; **China** is the dominant origin country
- 📍 **Product position** may influence sales volume — higher-placed items tend to sell more
- 📦 **Seasonal products** show distinct pricing behavior compared to year-round items

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| `Python 3` | Core programming language |
| `Pandas` | Data manipulation and aggregation |
| `NumPy` | Numerical operations |
| `Matplotlib` | Custom visualizations |
| `Seaborn` | Statistical visualizations |
| `SciPy` | Z-score outlier detection |
| `ydata-profiling` | Automated EDA report generation |
| `Jupyter Notebook` | Interactive analysis environment |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/zara-sales-eda.git
cd zara-sales-eda
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy ydata-profiling jupyter
```

### 3. Download the dataset
Download `Zara_sales_EDA.csv` from [Kaggle](https://www.kaggle.com/datasets/marixe/zara-sales-for-eda) and place it in the project folder.

### 4. Run the notebook
```bash
jupyter notebook Zara_Sales_EDA.ipynb
```

> ⚠️ The dataset uses **semicolon (`;`)** as a separator. The notebook already handles this with `pd.read_csv('Zara_sales_EDA.csv', sep=';')`.

---

## 👤 Author

**Esraa Mohamed**  
📅 February 2026

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
