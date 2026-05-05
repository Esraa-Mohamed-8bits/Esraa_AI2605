"""
Irrigation Need Predictor — Streamlit App
Run with:  streamlit run app.py

Only needs ONE file:  irrigation_bundle.json  (place next to app.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import matplotlib.pyplot as plt

#   Page config  
st.set_page_config(
    page_title="Irrigation Need Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_FILE = "irrigation_bundle.json"

PREDICTION_STYLES = {
    "Low":    ("#d4edda", "#155724", "Minimal irrigation required. Soil moisture is adequate."),
    "Medium": ("#fff3cd", "#856404", "Moderate irrigation recommended. Monitor closely."),
    "High":   ("#f8d7da", "#721c24",  "High irrigation urgently needed. Act immediately."),
}
DEFAULT_STYLE = ("#e2e3e5", "#383d41",  "Unknown prediction.")


#  Load everything from the single JSON bundle 
@st.cache_resource
def load_bundle():
    try:
        with open(BUNDLE_FILE) as f:
            bundle = json.load(f)

        # Rebuild LightGBM booster from string
        booster = lgb.Booster(model_str=bundle["lgbm_model_str"])

        # Rebuild imputer medians as numpy array
        medians = np.array(bundle["imputer_medians"])

        # Rebuild label encoder classes
        classes = bundle["label_encoder_classes"]

        return booster, medians, classes, bundle, None
    except Exception as e:
        return None, None, None, None, str(e)


booster, medians, classes, bundle, load_error = load_bundle()


#  Feature engineering (mirrors notebook) 
def engineer_features(df, num_cols):
    df = df.copy()
    c = df.columns.tolist()

    if "Temperature" in c and "Humidity" in c:
        df["heat_stress"]         = df["Temperature"] * (1 - df["Humidity"] / 100)
        df["temp_humidity_ratio"] = df["Temperature"] / (df["Humidity"] + 1e-6)

    if "Soil_Moisture" in c and "Temperature" in c:
        df["evapotransp_proxy"] = df["Soil_Moisture"] * df["Temperature"]
        df["soil_temp_ratio"]   = df["Soil_Moisture"] / (df["Temperature"] + 1e-6)

    if "Wind_Speed" in c and "Humidity" in c:
        df["moisture_loss_proxy"] = df["Wind_Speed"] * (100 - df["Humidity"])

    if "Rainfall" in c and "Soil_Moisture" in c:
        df["rainfall_deficit"]  = df["Soil_Moisture"] - df["Rainfall"]
        df["rainfall_soil_sum"] = df["Rainfall"] + df["Soil_Moisture"]

    if "Solar_Radiation" in c and "Humidity" in c:
        df["solar_humidity_ratio"] = df["Solar_Radiation"] / (df["Humidity"] + 1e-6)

    base = [col for col in num_cols if col in df.columns]
    df["row_mean"]  = df[base].mean(axis=1)
    df["row_std"]   = df[base].std(axis=1)
    df["row_range"] = df[base].max(axis=1) - df[base].min(axis=1)

    return df


#  Impute using saved medians 
def apply_imputer(df, feature_names):
    arr = df.values.astype(float)
    for i, median in enumerate(medians):
        col_nans = np.isnan(arr[:, i])
        arr[col_nans, i] = median
    return arr


#  Validation 
def validate_inputs(inputs):
    errors = []
    if not (-10 <= inputs.get("Temperature", 0) <= 60):
        errors.append("Temperature must be between -10°C and 60°C.")
    if not (0 <= inputs.get("Humidity", 50) <= 100):
        errors.append("Humidity must be between 0% and 100%.")
    if inputs.get("Soil_Moisture", 0) < 0:
        errors.append("Soil Moisture cannot be negative.")
    if inputs.get("Rainfall", 0) < 0:
        errors.append("Rainfall cannot be negative.")
    if inputs.get("Wind_Speed", 0) < 0:
        errors.append("Wind Speed cannot be negative.")
    return errors


#  Predict 
def predict(raw_inputs):
    num_cols      = bundle["num_cols"]
    feature_names = bundle["feature_names"]

    df = pd.DataFrame([raw_inputs])
    df = engineer_features(df, num_cols)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    arr   = apply_imputer(df, feature_names)
    proba = booster.predict(arr)[0]
    idx   = int(np.argmax(proba))
    label = classes[idx]
    return label, proba


#  Sidebar 
with st.sidebar:
    st.title("Irrigation Predictor")
    st.markdown("---")
    st.subheader("About")
    st.info(
        "This app uses machine learning to predict the **irrigation need** "
        "of a field based on environmental and soil conditions.\n\n"
        "**Model:** LightGBM (gradient boosting)\n\n"
        "**CV Balanced Accuracy:** ~95%+\n\n"
        "**Dataset:** Kaggle S6E4 — Playground Series 2025\n\n"
    )
    st.markdown("---")
    st.caption("Built with Streamlit · Powered by LightGBM")


#  Main 
st.title("Irrigation Need Predictor")
st.markdown("Enter field conditions below to get an instant irrigation recommendation.")

if load_error:
    st.error(
        f"Could not load model: `{load_error}`\n\n"
        f"Make sure **`{BUNDLE_FILE}`** is in the same folder as `app.py`."
    )
    st.stop()

tab1, tab2 = st.tabs(["Predict", " Batch Upload"])


# TAB 1 — Single Prediction
with tab1:
    st.subheader("Enter Field Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Atmosphere")
        temperature = st.slider("Temperature (°C)",  min_value=-10.0, max_value=60.0,   value=25.0, step=0.5)
        humidity    = st.slider("Humidity (%)",       min_value=0.0,   max_value=100.0,  value=60.0, step=1.0)
        rainfall    = st.slider("Rainfall (mm)",      min_value=0.0,   max_value=300.0,  value=10.0, step=0.5)

    with col2:
        st.markdown("##### Soil & Water")
        soil_moisture = st.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        wind_speed    = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=15.0, step=0.5)

    with col3:
        st.markdown("##### Solar")
        solar_rad = st.slider("Solar Radiation (W/m²)", min_value=0.0, max_value=1200.0, value=400.0, step=10.0)

    raw_inputs = {
        "Temperature":     temperature,
        "Humidity":        humidity,
        "Rainfall":        rainfall,
        "Soil_Moisture":   soil_moisture,
        "Wind_Speed":      wind_speed,
        "Solar_Radiation": solar_rad,
    }

    st.markdown("---")

    errors = validate_inputs(raw_inputs)
    if errors:
        for e in errors:
            st.error(f" {e}")
    else:
        label, proba = predict(raw_inputs)
        bg_color, text_color, meaning = PREDICTION_STYLES.get(str(label), DEFAULT_STYLE)

        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                padding: 24px 32px;
                border-radius: 12px;
                border-left: 6px solid {text_color};
                margin-bottom: 16px;
            ">
                <h2 style="margin:0; color:{text_color};">
                     Irrigation Need: <strong>{label}</strong>
                </h2>
                <p style="margin:8px 0 0; font-size:1.05em;">{meaning}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Prediction Confidence")
        prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability")

        fig, ax = plt.subplots(figsize=(6, 2.5))
        bar_colors = [
            "#d4edda" if c == "Low" else "#fff3cd" if c == "Medium" else "#f8d7da"
            for c in prob_df["Class"]
        ]
        bars = ax.barh(prob_df["Class"], prob_df["Probability"],
                       color=bar_colors, edgecolor="grey", linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, prob_df["Probability"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# TAB 2 — Batch Upload
with tab2:
    st.subheader(" Batch Prediction — Upload a CSV")
    st.markdown(
        "Upload a CSV file with the same columns as the training data. "
        "The app will predict irrigation need for every row."
    )

    with st.expander("Expected CSV columns"):
        st.code(", ".join(bundle["num_cols"] + bundle.get("cat_cols", [])))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch_df)} rows** and **{batch_df.shape[1]} columns**.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            missing = set(bundle["num_cols"]) - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: `{missing}`")
            else:
                feature_names = bundle["feature_names"]
                batch_eng = engineer_features(batch_df.copy(), bundle["num_cols"])
                for col in feature_names:
                    if col not in batch_eng.columns:
                        batch_eng[col] = 0.0
                batch_eng = batch_eng[feature_names]

                arr          = apply_imputer(batch_eng, feature_names)
                all_probas   = booster.predict(arr)
                preds_idx    = np.argmax(all_probas, axis=1)
                preds_labels = [classes[i] for i in preds_idx]

                batch_df["Predicted_Irrigation_Need"] = preds_labels

                st.markdown("##### Prediction Summary")
                summary    = batch_df["Predicted_Irrigation_Need"].value_counts()
                colors_map = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}
                bar_c      = [colors_map.get(c, "#cccccc") for c in summary.index]

                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.bar(summary.index, summary.values, color=bar_c, edgecolor="grey", linewidth=0.6)
                ax2.set_ylabel("Count")
                ax2.set_title("Batch Predictions Distribution", fontweight="bold")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                for i, (idx, val) in enumerate(summary.items()):
                    ax2.text(i, val + 0.5, str(val), ha="center", fontsize=10)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close()

                st.dataframe(
                    batch_df[list(bundle["num_cols"]) + ["Predicted_Irrigation_Need"]].head(20),
                    use_container_width=True,
                )

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Full Predictions CSV",
                    data=csv_out,
                    file_name="irrigation_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
