import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Zara Sales EDA", layout="wide")

#Auth.
USERNAME = "Esraa"
PASSWORD = "09876"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def logout():
    st.session_state.logged_in = False

#Login Page
def login():
    st.title("Login to access your EDAs app")
    username = st.text_input("Username",placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong username or password")

#Main App
def main_app():
    st.sidebar.button("Logout", on_click=logout)
    st.title("Zara Sales Dashboard")
    st.write("An Exploratory Data Analysis (EDA) on the Zara Sales dataset.")

    data = st.file_uploader("Upload your CSV file", type="csv")
    if data is None:
        st.warning("Please upload the Zara sales CSV file to get started.")
        return

    df = pd.read_csv(data)

    st.sidebar.header("Sections to explore")
    page = st.sidebar.selectbox("Go to:", [
        "1 · Dataset Preview",
        "2 · KPI Metrics",
        "3 · Price & Sales Distribution",
        "4 · Outlier Detection",
        "5 · Categorical Analysis",
        "6 · Sales by Season",
        "7 · Sales by Product Position",
        "8 · Promotion Impact",
        "9 · Scatter Plot — Price vs Sales",
        "10 · Pairplot Explorer",
        "11 · Correlation Heatmap",
        "12 · Top 10 Products by Revenue",
    ])

    # 1. Dataset Preview
    if page == "1 · Dataset Preview":
        st.subheader("Dataset Preview")
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())
        st.write(f"**Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
        st.write("**Column Types:**")
        st.dataframe(df.dtypes.rename("Type"))
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe(include="all"))

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            "The dataset contains **20,250 rows and 15 columns**. "
            "Numeric columns are `price`, `Sales Volume`, and `Revenue`. "
            "The rest are categorical (text) columns like `season`, `material`, `origin`, and `Promotion`. "
            "No critical missing values were found after cleaning — the data is ready for analysis."
        )

    # 2. KPI Metrics
    elif page == "2 · KPI Metrics":
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", f"{len(df):,}")
        col2.metric("Total Sales Volume", f"{int(df['Sales Volume'].sum()):,}")
        col3.metric("Average Price ($)", f"{df['price'].mean():.2f}")
        col4.metric("Total Revenue ($)", f"{df['Revenue'].sum():,.0f}")

        st.markdown("---")
        st.markdown("### 📝 Analysis")
        st.info(
            f"We have **{len(df):,} products** in total. "
            f"The average product price is **${df['price'].mean():.2f}**, "
            f"and the total estimated revenue across all products is **${df['Revenue'].sum():,.0f}**. "
            "These numbers give us a high-level picture of the dataset before going deeper."
        )

    # 3. Price & Sales Distribution
    elif page == "3 · Price & Sales Distribution":
        st.subheader("Price & Sales Distribution")

        fig1 = px.histogram(df, x="price", nbins=30,
                            title="Distribution of Product Prices",
                            color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Sales Volume", nbins=30,
                            title="Distribution of Sales Volume",
                            color_discrete_sequence=["coral"])
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"**Price:** Most products are priced between ${df['price'].quantile(0.25):.0f} and ${df['price'].quantile(0.75):.0f}. "
            f"The average price is ${df['price'].mean():.2f} but the median is ${df['price'].median():.2f}, "
            "which means the distribution is **right-skewed** — a few expensive products pull the average up.\n\n"
            f"**Sales Volume:** Most products sell between {int(df['Sales Volume'].quantile(0.25))} and {int(df['Sales Volume'].quantile(0.75))} units. "
            "Like price, it is also right-skewed — a small number of bestsellers sell far more than the rest."
        )

    # 4. Outlier Detection
    elif page == "4 · Outlier Detection":
        st.subheader("Outlier Detection — Boxplots")
        st.write(
            "A **boxplot** shows the spread of data. "
            "The box covers the middle 50% of values. "
            "The line inside is the median. "
            "Dots outside the whiskers are **outliers** — unusually high or low values."
        )

        tab1, tab2, tab3 = st.tabs(["Price", "Sales Volume", "Revenue"])

        def iqr_info(series, label):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            lower = Q1 - 1.5 * IQR
            n_out = ((series > upper) | (series < lower)).sum()
            return Q1, Q3, IQR, lower, upper, n_out

        with tab1:
            col = "price"
            Q1, Q3, IQR, lower, upper, n_out = iqr_info(df[col], col)
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name="Price", marker_color="steelblue",
                                  boxmean=True, boxpoints="outliers"))
            fig.update_layout(title="Price — Boxplot with Outliers",
                              yaxis_title="Price ($)", height=450)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q1 (25%)", f"${Q1:.2f}")
            c2.metric("Median", f"${df[col].median():.2f}")
            c3.metric("Q3 (75%)", f"${Q3:.2f}")
            c4.metric("Outliers found", f"{n_out:,}")

            st.markdown("---")
            st.markdown("### Analysis")
            st.info(
                f"The middle 50% of products are priced between **${Q1:.2f}** and **${Q3:.2f}** (IQR = ${IQR:.2f}). "
                f"There are **{n_out:,} outliers** — products priced above **${upper:.2f}**. "
                "These are likely premium or luxury Zara items. They are real products, not data errors, "
                "and they tell us that Zara has a small high-end range alongside its regular collection."
            )

        with tab2:
            col = "Sales Volume"
            Q1, Q3, IQR, lower, upper, n_out = iqr_info(df[col], col)
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name="Sales Volume", marker_color="coral",
                                  boxmean=True, boxpoints="outliers"))
            fig.update_layout(title="Sales Volume — Boxplot with Outliers",
                              yaxis_title="Units Sold", height=450)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q1 (25%)", f"{Q1:.0f}")
            c2.metric("Median", f"{df[col].median():.0f}")
            c3.metric("Q3 (75%)", f"{Q3:.0f}")
            c4.metric("Outliers found", f"{n_out:,}")

            st.markdown("---")
            st.markdown("### Analysis")
            st.info(
                f"Most products sell between **{Q1:.0f}** and **{Q3:.0f}** units. "
                f"There are **{n_out:,} outlier products** that sell more than **{upper:.0f}** units — these are Zara's bestsellers. "
                "A small number of products drive a very large portion of total sales volume, "
                "which is a common pattern in retail known as the **Pareto principle** (80/20 rule)."
            )

        with tab3:
            col = "Revenue"
            Q1, Q3, IQR, lower, upper, n_out = iqr_info(df[col], col)
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name="Revenue", marker_color="seagreen",
                                  boxmean=True, boxpoints="outliers"))
            fig.update_layout(title="Revenue — Boxplot with Outliers",
                              yaxis_title="Estimated Revenue ($)", height=450)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q1 (25%)", f"${Q1:,.0f}")
            c2.metric("Median", f"${df[col].median():,.0f}")
            c3.metric("Q3 (75%)", f"${Q3:,.0f}")
            c4.metric("Outliers found", f"{n_out:,}")

            st.markdown("---")
            st.markdown("###    Analysis")
            st.info(
                f"Revenue outliers — products earning more than **${upper:,.0f}** — are the real money-makers. "
                f"There are **{n_out:,}** such products. "
                "Interestingly, these aren't always the most expensive products — they're the ones that combine "
                "a decent price with very high sales volume. These products deserve focused attention from the business."
            )

    # 5. Categorical Analysis
    elif page == "5 · Categorical Analysis":
        st.subheader("Categorical Column Analysis")
        st.write("Select a categorical column to see the distribution of its values.")

        cat_cols = [c for c in ["Promotion", "Seasonal", "season", "section",
                                 "terms", "material", "origin", "Product Position"]
                    if c in df.columns]

        selected_col = st.selectbox("Choose a column:", cat_cols)
        counts = df[selected_col].value_counts().reset_index()
        counts.columns = [selected_col, "Count"]

        fig = px.bar(counts.head(20), x=selected_col, y="Count",
                     title=f"Distribution of '{selected_col}'",
                     color="Count",
                     color_continuous_scale="Blues")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(counts, use_container_width=True)

        st.markdown("---")
        st.markdown("### Analysis")

        top_val = counts.iloc[0][selected_col]
        top_count = counts.iloc[0]["Count"]
        total = counts["Count"].sum()
        top_pct = top_count / total * 100

        analyses = {
            "Promotion": f"About **{(df['Promotion']=='Yes').mean()*100:.1f}%** of products are on promotion. "
                         "This means the majority of Zara's catalog is sold at full price, "
                         "and promotions are used selectively rather than across the board.",
            "Seasonal": f"**{(df['Seasonal']=='Yes').mean()*100:.1f}%** of products are seasonal. "
                        "Seasonal items are tied to specific times of year and may have more pressure to sell quickly.",
            "season": f"The most common season is **{top_val}** with {top_count:,} products ({top_pct:.1f}%). "
                      "This tells us which seasons Zara focuses its catalog on most heavily.",
            "section": f"The dataset is split between MAN and WOMAN sections. "
                       f"**{top_val}** has the most products at {top_pct:.1f}%. "
                       "This reflects Zara's product targeting strategy across genders.",
            "terms": f"**{top_val}** is the most common product type, making up {top_pct:.1f}% of the catalog. "
                     "This shows which clothing categories Zara invests in most.",
            "material": f"**{top_val}** is the most used material ({top_pct:.1f}% of products). "
                        "Material choice affects cost, comfort, and target customer — "
                        "common materials like cotton and polyester keep production costs manageable.",
            "origin": f"**{top_val}** is the top manufacturing country, producing {top_pct:.1f}% of products. "
                      "Relying heavily on one country is a supply chain risk — "
                      "disruptions there would heavily impact Zara's inventory.",
            "Product Position": f"**{top_val}** is the most common store position with {top_pct:.1f}% of products. "
                                "Product placement in a store affects visibility and can directly influence sales.",
        }

        default_msg = (f"**{top_val}** is the most frequent value in '{selected_col}', "
                       f"appearing {top_count:,} times ({top_pct:.1f}% of all products).")

        st.info(analyses.get(selected_col, default_msg))

    # 6. Sales by Season
    elif page == "6 · Sales by Season":
        st.subheader("Sales Volume by Season")
        season_sales = df.groupby("season")["Sales Volume"].sum().reset_index().sort_values("Sales Volume", ascending=False)
        fig = px.bar(season_sales, x="season", y="Sales Volume", color="season",
                     title="Total Sales Volume by Season")
        st.plotly_chart(fig, use_container_width=True)

        top_season = season_sales.iloc[0]["season"]
        top_val = int(season_sales.iloc[0]["Sales Volume"])

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"**{top_season}** is the strongest season with **{top_val:,} total units sold**. "
            "This means Zara's most popular products are tied to this season. "
            "Understanding which seasons drive the most sales helps Zara plan inventory, "
            "marketing campaigns, and production schedules in advance."
        )

    # 7. Sales by Product Position
    elif page == "7 · Sales by Product Position":
        st.subheader("Sales by Product Position")
        fig = px.pie(df, names="Product Position", values="Sales Volume",
                     title="Sales Contribution by Store Position")
        st.plotly_chart(fig, use_container_width=True)

        top_pos = df.groupby("Product Position")["Sales Volume"].sum().idxmax()

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"Products in the **{top_pos}** position generate the highest share of total sales. "
            "This makes sense — certain store positions get more foot traffic and customer attention. "
            "Zara likely uses this insight to place high-margin or promotional products in the best spots."
        )

    # 8. Promotion Impact
    elif page == "8 · Promotion Impact":
        st.subheader("Promotion Impact on Sales")

        promo_sales = df.groupby("Promotion")["Sales Volume"].sum().reset_index()
        fig = px.bar(promo_sales, x="Promotion", y="Sales Volume", color="Promotion",
                     title="Total Sales Volume — Promoted vs Non-Promoted",
                     color_discrete_map={"Yes": "coral", "No": "steelblue"})
        st.plotly_chart(fig, use_container_width=True)

        promo_avg = df.groupby("Promotion")[["price", "Sales Volume", "Revenue"]].mean().round(2)
        st.write("**Average Price, Sales Volume & Revenue by Promotion:**")
        st.dataframe(promo_avg, use_container_width=True)

        avg_price_yes = promo_avg.loc["Yes", "price"] if "Yes" in promo_avg.index else 0
        avg_price_no  = promo_avg.loc["No",  "price"] if "No"  in promo_avg.index else 0
        avg_vol_yes   = promo_avg.loc["Yes", "Sales Volume"] if "Yes" in promo_avg.index else 0
        avg_vol_no    = promo_avg.loc["No",  "Sales Volume"] if "No"  in promo_avg.index else 0

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"Promoted products have an average price of **${avg_price_yes:.2f}** vs **${avg_price_no:.2f}** for non-promoted ones — "
            "confirming that promotions involve a price discount.\n\n"
            f"However, promoted products sell on average **{avg_vol_yes:.0f} units** vs **{avg_vol_no:.0f} units** for non-promoted ones. "
            "This is a classic **margin vs volume trade-off**: promotions reduce the price per item but increase the number of units sold. "
            "Whether this is profitable depends on Zara's production costs and profit margins."
        )

    # 9. Scatter Plot — Price vs Sales Volume
    elif page == "9 · Scatter Plot — Price vs Sales":
        st.subheader("Price vs. Sales Volume — Grouped by Promotion")
        st.write(
            "Each dot is a product. The x-axis shows its price and the y-axis shows how many units it sold. "
            "Colors separate promoted from non-promoted products."
        )

        fig = px.scatter(
            df, x="price", y="Sales Volume",
            color="Promotion",
            color_discrete_map={"Yes": "coral", "No": "steelblue"},
            opacity=0.45,
            title="Price vs. Sales Volume (colored by Promotion)",
            labels={"price": "Price ($)", "Sales Volume": "Units Sold"},
            hover_data=["name"] if "name" in df.columns else None
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

        corr = df["price"].corr(df["Sales Volume"])

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"The correlation between price and sales volume is **{corr:.2f}**, which is very close to zero. "
            "This tells us that **price alone does not determine how well a product sells** at Zara — "
            "cheap products don't necessarily sell more, and expensive ones don't necessarily sell less.\n\n"
            "Looking at the colors: **coral dots (promoted)** and **blue dots (non-promoted)** are spread across all price ranges, "
            "meaning Zara runs promotions on both cheap and expensive items. "
            "The scatter pattern is very wide with no clear trend — sales volume is likely driven by "
            "other factors like style, season, placement, and brand appeal."
        )

    # 10. Pairplot Explorer
    elif page == "10 · Pairplot Explorer":
        st.subheader("Pairplot Explorer")
        st.write(
            "Choose any two numeric columns to compare them in a scatter plot. "
            "You can also color the dots by a categorical column to spot patterns."
        )

        num_cols = [c for c in ["price", "Sales Volume", "Revenue"] if c in df.columns]
        cat_cols = [c for c in ["Promotion", "Seasonal", "season", "section"] if c in df.columns]

        col1, col2, col3 = st.columns(3)
        x_axis  = col1.selectbox("X-axis", num_cols, index=0)
        y_axis  = col2.selectbox("Y-axis", num_cols, index=1)
        hue_col = col3.selectbox("Color by", cat_cols, index=0)

        sample_df = df.sample(min(3000, len(df)), random_state=42)

        fig = px.scatter(
            sample_df, x=x_axis, y=y_axis, color=hue_col,
            opacity=0.5,
            title=f"{x_axis} vs {y_axis} — colored by {hue_col}",
            labels={x_axis: x_axis, y_axis: y_axis},
            hover_data=["name"] if "name" in df.columns else None
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

        corr = sample_df[x_axis].corr(sample_df[y_axis])

        st.markdown("---")
        st.markdown("### Analysis")
        if abs(corr) < 0.2:
            strength = "very weak — the two variables are almost unrelated"
        elif abs(corr) < 0.5:
            strength = "moderate — there is some relationship"
        else:
            strength = "strong — the two variables are closely linked"

        direction = "positive (both go up together)" if corr > 0 else "negative (one goes up as the other goes down)"

        st.info(
            f"The correlation between **{x_axis}** and **{y_axis}** is **{corr:.2f}**, which is {strength}. "
            f"The relationship is {direction}.\n\n"
            f"The color grouping by **{hue_col}** helps reveal whether different groups behave differently. "
            "If the colored clusters overlap heavily, the grouping variable doesn't change the relationship much. "
            "If they separate, that variable is an important factor."
        )

    # 11. Correlation Heatmap
    elif page == "11 · Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        st.write(
            "A heatmap shows how strongly pairs of numeric columns are related. "
            "Values close to **1** mean a strong positive relationship, "
            "close to **-1** mean a strong negative relationship, "
            "and close to **0** mean no relationship."
        )

        numeric_df = df[["price", "Sales Volume", "Revenue"]]
        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f",
                    linewidths=0.5, vmin=-1, vmax=1)
        st.pyplot(fig)

        pr_corr  = corr_matrix.loc["price", "Revenue"]
        pv_corr  = corr_matrix.loc["price", "Sales Volume"]
        vr_corr  = corr_matrix.loc["Sales Volume", "Revenue"]

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"**Price ↔ Revenue: {pr_corr:.2f}** — {'Strong positive' if pr_corr > 0.5 else 'Moderate' if pr_corr > 0.2 else 'Weak'} relationship. "
            "Higher-priced products tend to generate more revenue (since Revenue = Price × Volume).\n\n"
            f"**Price ↔ Sales Volume: {pv_corr:.2f}** — Very weak relationship. "
            "Price doesn't strongly predict how many units are sold — demand at Zara is not simply price-driven.\n\n"
            f"**Sales Volume ↔ Revenue: {vr_corr:.2f}** — "
            "{'Strong positive' if vr_corr > 0.5 else 'Moderate'} relationship. "
            "Products that sell more units naturally generate more revenue, as expected."
        )

    # 12. Top 10 Products by Revenue
    elif page == "12 · Top 10 Products by Revenue":
        st.subheader("Top 10 Products by Estimated Revenue")

        top10 = df.nlargest(10, "Revenue")[["name", "price", "Sales Volume", "Revenue"]].reset_index(drop=True)
        st.dataframe(top10, use_container_width=True)

        fig = px.bar(top10, x="Revenue", y="name", orientation="h",
                     title="Top 10 Products by Estimated Revenue",
                     color="Revenue", color_continuous_scale="YlOrBr")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        top_product = top10.iloc[0]["name"]
        top_rev     = top10.iloc[0]["Revenue"]
        top_price   = top10.iloc[0]["price"]
        top_vol     = top10.iloc[0]["Sales Volume"]

        st.markdown("---")
        st.markdown("### Analysis")
        st.info(
            f"The top revenue product is **{top_product}**, earning an estimated **${top_rev:,.0f}** "
            f"(priced at ${top_price:.2f} × {top_vol:,} units sold).\n\n"
            "Notice that the highest-revenue products are not always the most expensive — "
            "they are the ones with the best combination of price and sales volume. "
            "This confirms that **volume matters just as much as price** when it comes to revenue. "
            "Zara should prioritize restocking and promoting these top performers."
        )


#Router
if st.session_state.logged_in:
    main_app()
else:
    login()