"""
House Price Prediction - Streamlit Frontend
Run with:  streamlit run app.py
Backend  : pandas + scikit-learn model (trained in the notebook, loaded with joblib)
Frontend : Streamlit + Matplotlib / Seaborn
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "house_price_regression_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
REFERENCE_YEAR = 2026
FEATURES = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Lot_Size",
            "Garage_Size", "Neighborhood_Quality", "House_Age"]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ backend
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def prepare(df):
    data = df.copy()
    data["House_Age"] = REFERENCE_YEAR - data["Year_Built"]
    return data[FEATURES], data["House_Price"]


@st.cache_resource
def load_model():
    """Load the model saved by the notebook. If it is missing or incompatible
    with the installed scikit-learn version, retrain quickly from the CSV."""
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], False
    except Exception:
        X, y = prepare(load_data())
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.01))])
        model.fit(X_tr, y_tr)
        return model, True


def load_metrics(model):
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    X, y = prepare(load_data())
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    p = model.predict(X_te)
    return {"best_model": "Ridge Regression", "r2": float(r2_score(y_te, p)),
            "mae": float(mean_absolute_error(y_te, p)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, p))),
            "n_train": len(X) - len(X_te), "n_test": len(X_te),
            "comparison": [], "feature_importance": {}}


def predict(model, sqft, beds, baths, year_built, lot, garage, quality):
    row = pd.DataFrame([{
        "Square_Footage": sqft, "Num_Bedrooms": beds, "Num_Bathrooms": baths,
        "Lot_Size": lot, "Garage_Size": garage,
        "Neighborhood_Quality": quality, "House_Age": REFERENCE_YEAR - year_built}])[FEATURES]
    return float(model.predict(row)[0])


df = load_data()
model, retrained = load_model()
metrics = load_metrics(model)

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🏠 House Price Predictor")
page = st.sidebar.radio("Navigate", ["🔮 Predict Price", "📊 Data Explorer",
                                     "📈 Model Performance", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.caption("IBM SkillsBuild Internship Project")
if retrained:
    st.sidebar.info("Saved model not found/compatible - model was retrained from the CSV.")

# --------------------------------------------------------------- page 1
if page == "🔮 Predict Price":
    st.title("🔮 Predict House Price")
    st.write("Enter the property details and click **Predict** to get an estimated price.")

    c1, c2, c3 = st.columns(3)
    with c1:
        sqft = st.number_input("Square footage", 500, 6000, 2500, step=50)
        beds = st.slider("Bedrooms", 1, 6, 3)
        baths = st.slider("Bathrooms", 1, 4, 2)
    with c2:
        year_built = st.slider("Year built", 1950, REFERENCE_YEAR, 2005)
        lot = st.number_input("Lot size (acres)", 0.1, 6.0, 2.5, step=0.1)
        garage = st.slider("Garage spaces", 0, 3, 1)
    with c3:
        quality = st.slider("Neighborhood quality (1-10)", 1, 10, 7)

    if st.button("💰 Predict Price", type="primary"):
        price = predict(model, sqft, beds, baths, year_built, lot, garage, quality)
        st.success("Prediction complete")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", f"{price:,.0f}")
        m2.metric("Likely range (±RMSE)", f"{price - metrics['rmse']:,.0f} - {price + metrics['rmse']:,.0f}")
        m3.metric("Price per sq ft", f"{price / sqft:,.1f}")

        avg = df["House_Price"].mean()
        diff = (price - avg) / avg * 100
        st.write(f"This is **{abs(diff):.1f}% {'above' if diff >= 0 else 'below'}** the dataset's average price ({avg:,.0f}).")

        fig, ax = plt.subplots(figsize=(8, 3.6))
        sns.histplot(df["House_Price"], bins=30, color="#9DC3E6", ax=ax)
        ax.axvline(price, color="red", linewidth=2, label="Your house")
        ax.axvline(avg, color="black", linestyle="--", label="Average")
        ax.set_xlabel("House price")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Price sensitivity to size")
        sizes = np.arange(500, 5001, 250)
        preds = [predict(model, s, beds, baths, year_built, lot, garage, quality) for s in sizes]
        fig2, ax2 = plt.subplots(figsize=(8, 3.6))
        ax2.plot(sizes, preds, marker="o", color="#2E75B6")
        ax2.scatter([sqft], [price], color="red", zorder=5, label="Your house")
        ax2.set_xlabel("Square footage")
        ax2.set_ylabel("Predicted price")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

# --------------------------------------------------------------- page 2
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Houses", f"{len(df):,}")
    k2.metric("Average price", f"{df['House_Price'].mean():,.0f}")
    k3.metric("Min price", f"{df['House_Price'].min():,.0f}")
    k4.metric("Max price", f"{df['House_Price'].max():,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Distributions", "Correlation", "Relationships"])
    with tab1:
        st.dataframe(df, use_container_width=True)
        st.write("Summary statistics")
        st.dataframe(df.describe().T.round(2), use_container_width=True)
    with tab2:
        col = st.selectbox("Select a column", df.columns, index=len(df.columns) - 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col], kde=True, color="#2E75B6", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with tab4:
        x_col = st.selectbox("X axis", [c for c in df.columns if c != "House_Price"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=df, x=x_col, y="House_Price", alpha=0.6, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------------------- page 3
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write(f"**Final model:** {metrics['best_model']}")
    a, b, c = st.columns(3)
    a.metric("R² score", f"{metrics['r2']:.4f}")
    b.metric("MAE", f"{metrics['mae']:,.0f}")
    c.metric("RMSE", f"{metrics['rmse']:,.0f}")

    if metrics.get("comparison"):
        st.subheader("Model comparison")
        comp = pd.DataFrame(metrics["comparison"])
        st.dataframe(comp, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=comp, x="Test R2", y="Model", color="#2E75B6", ax=ax)
        ax.set_xlim(0, 1.05)
        st.pyplot(fig)
        plt.close(fig)

    if metrics.get("feature_importance"):
        st.subheader("Feature importance")
        imp = pd.Series(metrics["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        imp.plot(kind="barh", color="#2E75B6", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    X, y = prepare(df)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = model.predict(X_te)
    st.subheader("Actual vs predicted (test set)")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_te, pred, alpha=0.6, color="#2E75B6")
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------- page 4
else:
    st.title("ℹ️ About this project")
    st.markdown("""
**House Price Prediction** - IBM SkillsBuild Internship project.

* **Dataset:** 1,000 houses, 7 input features, target = `House_Price`
* **Backend:** Python, pandas, NumPy, scikit-learn (Ridge / Random Forest / Gradient Boosting compared)
* **Frontend:** Streamlit, Matplotlib, Seaborn
* **Pipeline:** cleaning check → EDA → feature engineering (House_Age) → scaling → model selection → tuning → deployment

Prices are estimates learned from the sample dataset and are for learning purposes only.
""")
