# =========================================================
# ML MODEL BEHAVIOUR INSPECTOR — FINAL STABLE VERSION
# =========================================================

# -------------------------
# IMPORTS
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

import google.generativeai as genai

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="ML Model Behaviour Inspector",
    layout="wide"
)

st.title("🧠 ML Model Behaviour Inspector")
st.caption("Understand how your ML model behaves — not just its accuracy")

# -------------------------
# GEMINI CONFIG
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# SESSION STATE
# -------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

# =========================================================
# SIDEBAR — DATA UPLOAD
# =========================================================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)

if "df_original" not in st.session_state:
    st.session_state.df_original = df.copy()
    st.session_state.df_cleaned = df.copy()

df_curr = st.session_state.df_cleaned.copy()

st.subheader("Dataset Preview (first 5 rows)")
st.dataframe(df_curr.head())

# =========================================================
# TARGET SELECTION
# =========================================================
st.sidebar.header("🎯 Target Variable")

target_col = st.sidebar.selectbox("Select target column", df_curr.columns)

X = df_curr.drop(columns=[target_col])
y = df_curr[target_col]

X = pd.get_dummies(X, drop_first=True)

# =========================================================
# PROBLEM TYPE
# =========================================================
st.sidebar.header("🧩 Problem Type")

problem_type = st.sidebar.radio(
    "Choose problem type",
    ["Auto", "Regression", "Classification"]
)

target_is_categorical = y.dtype == "object"

if problem_type == "Auto":
    is_classification = target_is_categorical
elif problem_type == "Classification":
    is_classification = True
else:
    is_classification = False

if target_is_categorical and not is_classification:
    st.error("❌ Target is categorical. Choose Classification.")
    st.stop()

if not target_is_categorical and is_classification:
    st.warning("⚠ Numeric target usually implies Regression.")

if is_classification:
    y = LabelEncoder().fit_transform(y)

# =========================================================
# DATA SPLIT & K-FOLD
# =========================================================
st.sidebar.header("🔀 Data Split & Validation")

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
use_kfold = st.sidebar.checkbox("Use K-Fold CV")
k_folds = st.sidebar.slider("K (folds)", 3, 10, 5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# =========================================================
# MODEL SELECTION
# =========================================================
st.sidebar.header("🤖 Model")

if is_classification:
    model_name = st.sidebar.selectbox(
        "Select model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )
    model = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }[model_name]
else:
    model_name = st.sidebar.selectbox(
        "Select model",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )
    model = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }[model_name]

if use_kfold:
    score = cross_val_score(
        model, X, y, cv=k_folds,
        scoring="accuracy" if is_classification else "r2"
    ).mean()
    st.sidebar.write(f"Mean CV Score: {score:.3f}")

# =========================================================
# TRAIN / SAVE / LOAD
# =========================================================
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Train Model"):
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.trained = True
    st.success("Model trained successfully!")

if st.sidebar.button("💾 Save Model") and st.session_state.trained:
    joblib.dump(st.session_state.model, "trained_model.pkl")
    st.sidebar.success("Model saved")

if st.sidebar.button("📂 Load Model"):
    try:
        st.session_state.model = joblib.load("trained_model.pkl")
        st.session_state.trained = True
        st.sidebar.success("Model loaded")
    except:
        st.sidebar.error("No saved model found")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Visuals", "📈 Metrics", "🧠 Behaviour", "🔍 Clustering", "🤖 Gemini", "🧼 Data Quality"]
)

# =========================================================
# DATA QUALITY TAB
# =========================================================
with tab6:
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_curr.shape[0])
    col2.metric("Columns", df_curr.shape[1])
    col3.metric("Missing Values", df_curr.isnull().sum().sum())
    col4.metric("Missing %", round(df_curr.isnull().mean().mean()*100, 2))

    if st.checkbox("Show full dataset"):
        st.dataframe(df_curr)

# =========================================================
# STOP IF NOT TRAINED
# =========================================================
if not st.session_state.trained:
    st.warning("Train model to view analysis")
    st.stop()

model = st.session_state.model
y_pred = model.predict(X_test)

# =========================================================
# VISUALS
# =========================================================
with tab1:
    feature = st.selectbox("Select feature", X.columns)
    st.plotly_chart(px.histogram(df_curr, x=feature), use_container_width=True)
    st.plotly_chart(px.box(df_curr, y=feature), use_container_width=True)

# =========================================================
# METRICS
# =========================================================
with tab2:
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", round(acc, 3))
        st.text(classification_report(y_test, y_pred))
        st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.metric("RMSE", round(rmse, 3))
        st.metric("R²", round(r2, 3))

# =========================================================
# BEHAVIOUR (IMPORTANCE + SHAP)
# =========================================================
with tab3:
    st.subheader("Feature Importance")

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if importance is not None:
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.plotly_chart(
            px.bar(imp_df, x="Importance", y="Feature", orientation="h"),
            use_container_width=True
        )

    st.subheader("SHAP Explainability")

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        try:
            X_shap = X_test.sample(min(200, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            fig = plt.figure()
            shap.summary_plot(
                shap_values[1] if isinstance(shap_values, list) else shap_values,
                X_shap,
                show=False
            )
            st.pyplot(fig)
        except:
            st.warning("SHAP could not be computed")
    else:
        st.info("SHAP available only for tree models")

# =========================================================
# CLUSTERING
# =========================================================
with tab4:
    k = st.slider("Number of clusters", 2, 10, 3)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)

    cluster_df = df_curr.copy()
    cluster_df["Cluster"] = labels

    st.bar_chart(cluster_df["Cluster"].value_counts())

    st.subheader("Cluster Profiles")
    st.dataframe(cluster_df.groupby("Cluster").mean(numeric_only=True))

    fx = st.selectbox("X-axis", X.columns)
    fy = st.selectbox("Y-axis", X.columns, index=1)

    st.plotly_chart(
        px.scatter(cluster_df, x=fx, y=fy, color="Cluster"),
        use_container_width=True
    )

# =========================================================
# GEMINI
# =========================================================
with tab5:
    st.subheader("Ask Gemini for ML Advice")

    summary = f"""
Model: {model_name}
Problem Type: {'Classification' if is_classification else 'Regression'}
Samples: {X.shape[0]}
Features: {X.shape[1]}
Missing Values: {df_curr.isnull().sum().sum()}
"""

    st.text_area("Summary", summary, height=150)

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set")
    else:
        if st.button("✨ Ask Gemini"):
            with st.spinner("Gemini analyzing..."):
                gmodel = genai.GenerativeModel("gemini-1.5-pro")
                response = gmodel.generate_content(
                    "You are an ML expert. Analyze and suggest improvements:\n" + summary
                )
            st.success(response.text)
