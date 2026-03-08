# 🛍️ Zara Sales EDA Dashboard

**Author:** Esraa Mohamed
**Date:** February 2026
**Dataset:** [Zara Sales for EDA — Kaggle](https://www.kaggle.com/datasets/marixe/zara-sales-for-eda)

---

## 📌 About This Project

This is an interactive **Exploratory Data Analysis (EDA) dashboard** built with **Streamlit** for the Zara Sales dataset.

The goal of this project was to go beyond automated profiling tools like **ydata-profiling** and build something **more interactive, more readable, and more beginner-friendly**. Instead of dumping a static HTML report on the user, this app lets you explore the data step by step — choosing your own columns, switching between views, and reading plain-English analysis after every chart.

> 💡 **Why not just use ydata-profiling?**
> ydata-profiling generates a report automatically, but it gives you a wall of information all at once with no control over what you see. This dashboard was built as a **better alternative** — you can navigate section by section, interact with the charts, pick your own axes for the pairplot, and get a written explanation of what each visualization actually means. It's designed for people learning data analysis, not just people who already know what they're looking at.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install streamlit pandas plotly seaborn matplotlib numpy
```

### 2. Run the app

```bash
streamlit run zara_eda_app.py
```

### 3. Login

| Field    | Value   |
|----------|---------|
| Username | `Esraa` |
| Password | `09876` |

### 4. Upload the dataset

Upload `cleaned_zara_sales.csv` when prompted inside the app.

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `zara_eda_app.py` | Main Streamlit application |
| `cleaned_zara_sales.csv` | Cleaned dataset used in the app |
| `Zara_Sales_EDA.ipynb` | Original Jupyter notebook EDA |
| `zara_sales_eda_report.html` | Auto-generated ydata-profiling report (for comparison) |
| `README.md` | This file |

---

## 📊 Dashboard Sections

The app is divided into **12 sections**, each with a chart and a written analysis:

| # | Section | What You'll See |
|---|---------|-----------------|
| 1 | **Dataset Preview** | First rows, column types, statistical summary |
| 2 | **KPI Metrics** | Total products, sales volume, average price, revenue |
| 3 | **Price & Sales Distribution** | Histograms showing how price and sales are spread |
| 4 | **Outlier Detection** | Boxplots with tabs for Price, Sales Volume, and Revenue — with IQR stats and explanations |
| 5 | **Categorical Analysis** | Dropdown to pick any categorical column and see its bar chart |
| 6 | **Sales by Season** | Which season drives the most sales |
| 7 | **Sales by Product Position** | Pie chart of sales by store placement |
| 8 | **Promotion Impact** | How promotions affect price and sales volume |
| 9 | **Scatter Plot** | Price vs. Sales Volume grouped by Promotion status |
| 10 | **Pairplot Explorer** | Pick any two numeric columns and a color grouping — scatter updates live |
| 11 | **Correlation Heatmap** | How price, sales volume, and revenue relate to each other |
| 12 | **Top 10 by Revenue** | The highest-earning products with a horizontal bar chart |

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Web app framework |
| `Pandas` | Data loading and manipulation |
| `Plotly` | Interactive charts |
| `Seaborn` + `Matplotlib` | Correlation heatmap |
| `NumPy` | Numerical calculations (IQR, outliers) |

---

## 📁 Dataset Overview

The dataset contains **20,250 rows** and **15 columns** after cleaning:

| Column | Type | Description |
|--------|------|-------------|
| `Product Position` | Categorical | Store placement (Aisle, End-cap, etc.) |
| `Promotion` | Categorical | Whether the product is on promotion (Yes/No) |
| `Seasonal` | Categorical | Whether it's a seasonal product (Yes/No) |
| `Sales Volume` | Numeric | Number of units sold |
| `name` | Text | Product name |
| `description` | Text | Product description |
| `price` | Numeric | Product price in USD |
| `terms` | Categorical | Product type (jackets, shirts, etc.) |
| `section` | Categorical | MAN or WOMAN |
| `season` | Categorical | Winter, Summer, Spring, Autumn |
| `material` | Categorical | Main fabric material |
| `origin` | Categorical | Manufacturing country |
| `Revenue` | Numeric | Estimated revenue (price × sales volume) |

---

## 🔍 Key Findings

- **Price and Sales Volume are almost uncorrelated** — cheap products don't always sell more at Zara.
- **Promoted products** have lower average prices but higher sales volume — a classic margin vs. volume trade-off.
- **A small number of bestsellers** drive a disproportionate share of total sales (Pareto principle).
- **China** is the dominant manufacturing origin — a potential supply chain risk.
- **Cotton and Polyester** are the most common materials.
- **Outliers in price** represent real premium products; outliers in sales volume are Zara's top sellers — neither should be removed.

---

## 🙏 Acknowledgements

- Dataset sourced from [Kaggle — Zara Sales for EDA](https://www.kaggle.com/datasets/marixe/zara-sales-for-eda)
- Built as part of a data analysis learning project