# =========================================================
# ML MODEL BEHAVIOUR INSPECTOR — FINAL FIXED VERSION
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
# PAGE CONFIG (MUST BE FIRST)
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
# SESSION STATE INIT
# -------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

# =========================================================
# SIDEBAR — DATA UPLOAD
# =========================================================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file", type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)

# Initialize dataset states ONCE
if "df_original" not in st.session_state:
    st.session_state.df_original = df.copy()
    st.session_state.df_cleaned = df.copy()

st.subheader("Dataset Preview (first 5 rows)")
st.dataframe(df.head())

df_curr = st.session_state.df_cleaned.copy()

# =========================================================
# SIDEBAR — TARGET SELECTION
# =========================================================
st.sidebar.header("🎯 Target Variable")

target_col = st.sidebar.selectbox(
    "Select target column",
    df_curr.columns
)

X = df_curr.drop(columns=[target_col])
y = df_curr[target_col]

# Encode categorical FEATURES
X = pd.get_dummies(X, drop_first=True)

# =========================================================
# SIDEBAR — PROBLEM TYPE
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

# Validation
if target_is_categorical and not is_classification:
    st.error(
        "❌ Selected target is categorical.\n"
        "Please choose **Classification**."
    )
    st.stop()

if not target_is_categorical and is_classification:
    st.warning(
        "⚠ Target is numeric.\n"
        "Classification is usually meant for categorical targets."
    )

# Encode TARGET if classification
if is_classification:
    le = LabelEncoder()
    y = le.fit_transform(y)

# =========================================================
# SIDEBAR — DATA SPLIT & K-FOLD
# =========================================================
st.sidebar.header("🔀 Data Split & Validation")

test_size = st.sidebar.slider(
    "Test size",
    0.1, 0.5, 0.2, 0.05
)

use_kfold = st.sidebar.checkbox("Use K-Fold Cross Validation")
k_folds = st.sidebar.slider("K (folds)", 3, 10, 5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# =========================================================
# SIDEBAR — MODEL SELECTION
# =========================================================
st.sidebar.header("🤖 Model")

if is_classification:
    model_name = st.sidebar.selectbox(
        "Select model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier()
else:
    model_name = st.sidebar.selectbox(
        "Select model",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()

# K-Fold preview
if use_kfold:
    if is_classification:
        scores = cross_val_score(model, X, y, cv=k_folds, scoring="accuracy")
        st.sidebar.write(f"Mean CV Accuracy: {scores.mean():.3f}")
    else:
        scores = cross_val_score(model, X, y, cv=k_folds, scoring="r2")
        st.sidebar.write(f"Mean CV R²: {scores.mean():.3f}")

# =========================================================
# SIDEBAR — TRAIN / SAVE / LOAD
# =========================================================
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Train Model"):
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.trained = True
    st.success("Model trained successfully!")

if st.sidebar.button("💾 Save Model"):
    if st.session_state.trained:
        joblib.dump(st.session_state.model, "trained_model.pkl")
        st.sidebar.success("Model saved")
    else:
        st.sidebar.warning("Train model first")

if st.sidebar.button("📂 Load Model"):
    try:
        st.session_state.model = joblib.load("trained_model.pkl")
        st.session_state.trained = True
        st.sidebar.success("Model loaded")
    except:
        st.sidebar.error("No saved model found")

# =========================================================
# MAIN TABS (ALWAYS SHOWN)
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Visuals", "📈 Metrics", "🧠 Behaviour", "🔍 Clustering", "🤖 Recommendations", "🧼 Data Quality"]
)

# =========================================================
# TAB 6 — DATA QUALITY (AVAILABLE ALWAYS)
# =========================================================
with tab6:
    st.subheader("📄 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_curr.shape[0])
    col2.metric("Columns", df_curr.shape[1])
    col3.metric("Missing Values", df_curr.isnull().sum().sum())
    col4.metric(
        "Missing %",
        round((df_curr.isnull().sum().sum() / df_curr.size) * 100, 2)
    )

    if st.checkbox("Show full dataset"):
        st.dataframe(df_curr)

    st.markdown("---")
    st.subheader("🧩 Handle Missing Values")

    strategy = st.selectbox(
        "Filling strategy",
        ["Mean", "Median", "Mode", "Drop Rows"]
    )

    numeric_cols = df_curr.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.multiselect(
        "Columns (leave empty = all numeric)",
        numeric_cols
    )

    if st.button("Apply Missing Value Strategy"):
        df_new = df_curr.copy()
        cols = selected_cols if selected_cols else numeric_cols

        for col in cols:
            if strategy == "Mean":
                df_new[col] = df_new[col].fillna(df_new[col].mean())
            elif strategy == "Median":
                df_new[col] = df_new[col].fillna(df_new[col].median())
            elif strategy == "Mode":
                df_new[col] = df_new[col].fillna(df_new[col].mode()[0])
            elif strategy == "Drop Rows":
                df_new = df_new.dropna()

        st.session_state.df_cleaned = df_new
        st.session_state.trained = False
        st.success("Missing values handled. Please retrain the model.")

# =========================================================
# STOP ANALYSIS IF NOT TRAINED
# =========================================================
if not st.session_state.trained:
    st.warning("Train a model to view analysis tabs.")
    st.stop()

model = st.session_state.model
y_pred = model.predict(X_test)

# =========================================================
# TAB 1 — VISUALS
# =========================================================
with tab1:
    feature = st.selectbox("Select feature", X.columns)
    st.plotly_chart(px.histogram(df_curr, x=feature), use_container_width=True)
    st.plotly_chart(px.box(df_curr, y=feature), use_container_width=True)

# =========================================================
# TAB 2 — METRICS
# =========================================================
with tab2:
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", round(acc, 3))
        st.text(classification_report(y_test, y_pred))
        st.plotly_chart(
            px.imshow(confusion_matrix(y_test, y_pred), text_auto=True),
            use_container_width=True
        )
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.metric("RMSE", round(rmse, 3))
        st.metric("R²", round(r2, 3))

# =========================================================
# TAB 3 — BEHAVIOUR
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
    else:
        st.info("Feature importance not available for this model")

    st.subheader("SHAP Explainability")
    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig = plt.figure()
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_test,
            show=False
        )
        st.pyplot(fig)
    else:
        st.info("SHAP available only for tree-based models")

# =========================================================
# TAB 4 — CLUSTERING
# =========================================================
with tab4:
    k = st.slider("Number of clusters", 2, 10, 3)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)

    cluster_df = df_curr.copy()
    cluster_df["Cluster"] = labels

    st.bar_chart(cluster_df["Cluster"].value_counts())

    fx = st.selectbox("X-axis", X.columns)
    fy = st.selectbox("Y-axis", X.columns, index=1)

    st.plotly_chart(
        px.scatter(cluster_df, x=fx, y=fy, color="Cluster"),
        use_container_width=True
    )

# =========================================================
# TAB 5 — GEMINI
# =========================================================
with tab5:
    st.subheader("Ask Gemini for Model Advice")

    summary = f"""
Model: {model_name}
Problem Type: {'Classification' if is_classification else 'Regression'}
Samples: {X.shape[0]}
Features: {X.shape[1]}
Missing Values: {df_curr.isnull().sum().sum()}
"""

    st.text_area("Summary sent to Gemini", summary, height=150)

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set")
    else:
        if st.button("✨ Ask Gemini"):
            with st.spinner("Gemini analyzing..."):
                gmodel = genai.GenerativeModel("gemini-1.5-flash")
                response = gmodel.generate_content(
                    "Give ML improvement advice based on:\n" + summary
                )
            st.success(response.text)
