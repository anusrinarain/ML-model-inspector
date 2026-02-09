
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn
import google.generativeai as genai

'''genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
st.write("Gemini key loaded:", os.getenv("GEMINI_API_KEY") is not None)'''

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="ML Model Behaviour Inspector",
    layout="wide"
)

st.title("🧠 ML Model Behaviour Inspector")
st.caption("Understand how your ML model behaves — not just its accuracy")

#GEMINI CONFIG
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ================================
# SESSION STATE INIT
# ================================
if "trained" not in st.session_state:
    st.session_state.trained = False

# ================================
# SIDEBAR — DATA UPLOAD
# ================================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file", type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(df.head())
if "df_original" not in st.session_state:
    st.session_state.df_original = df.copy()
    st.session_state.df_cleaned = df.copy()


# ================================
# TARGET SELECTION
# ================================
st.sidebar.header("🎯 Target Variable")

target_col = st.sidebar.selectbox(
    "Select target column",
    df.columns
)

df_curr = st.session_state.df_cleaned.copy()

X = df_curr.drop(columns=[target_col])
y = df_curr[target_col]

# Encode categorical FEATURES
X = pd.get_dummies(X, drop_first=True)

# =====================================================
# SIDEBAR — PROBLEM TYPE
# =====================================================
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
        "❌ Target variable is categorical.\n"
        "Please choose **Classification**."
    )
    st.stop()

if not target_is_categorical and is_classification:
    st.warning(
        "⚠ Target variable is numeric.\n"
        "Classification is usually meant for categorical targets."
    )

# Encode TARGET if classification
if is_classification:
    le = LabelEncoder()
    y = le.fit_transform(y)

# =====================================================
# SIDEBAR — DATA SPLIT & CV
# =====================================================
st.sidebar.header("🔀 Data Split & Validation")

test_size = st.sidebar.slider(
    "Test size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

use_kfold = st.sidebar.checkbox("Use K-Fold Cross Validation")
k_folds = st.sidebar.slider(
    "K (folds)",
    min_value=3,
    max_value=10,
    value=5
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# =====================================================
# SIDEBAR — MODEL SELECTION
# =====================================================
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


# =====================================================
# SIDEBAR — TRAIN / SAVE / LOAD
# =====================================================
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Train Model"):
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.trained = True
    st.success("Model trained successfully!")

if st.sidebar.button("💾 Save Model"):
    if st.session_state.trained:
        joblib.dump(st.session_state.model, "trained_model.pkl")
        st.sidebar.success("Model saved!")
    else:
        st.sidebar.warning("Train a model first.")

if st.sidebar.button("📂 Load Model"):
    try:
        st.session_state.model = joblib.load("trained_model.pkl")
        st.session_state.trained = True
        st.sidebar.success("Model loaded!")
    except:
        st.sidebar.error("No saved model found.")
# =====================================================
# STOP IF NOT TRAINED
# =====================================================
if not st.session_state.trained:
    st.warning("Train a model to view analysis.")
    st.stop()

model = st.session_state.model
y_pred = model.predict(X_test)



# ================================
# TRAIN BUTTON (CRITICAL)
# ================================
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Train Model"):
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.trained = True
    st.success("Model trained successfully!")
if st.sidebar.button("💾 Save Trained Model"):
    joblib.dump(model, "trained_model.pkl")
    st.sidebar.success("Model saved successfully!")
if st.sidebar.button("📂 Load Saved Model"):
    model = joblib.load("trained_model.pkl")
    st.session_state.model = model
    st.session_state.trained = True
    st.sidebar.success("Model loaded successfully!")
st.markdown("---")
st.subheader("🔍 Missing Value Detection")

if st.button("Detect Missing Values"):
    null_df = pd.DataFrame({
        "Column": df_curr.columns,
        "Missing Count": df_curr.isnull().sum(),
        "Missing %": (df_curr.isnull().mean() * 100).round(2)
    }).sort_values("Missing Count", ascending=False)

    st.dataframe(null_df)

    if null_df["Missing Count"].sum() == 0:
        st.success("No missing values detected 🎉")
    else:
        st.warning("Dataset contains missing values.")

st.markdown("---")
st.subheader("🧩 Handle Missing Values")

strategy = st.selectbox(
    "Choose filling strategy",
    ["Mean", "Median", "Mode", "Drop Rows"]
)

numeric_cols = df_curr.select_dtypes(include=np.number).columns.tolist()

selected_cols = st.multiselect(
    "Select columns (leave empty = all numeric columns)",
    numeric_cols
)
if st.button("Apply Missing Value Strategy"):
    df_new = df_curr.copy()

    cols_to_use = selected_cols if selected_cols else numeric_cols

    if strategy == "Mean":
        for col in cols_to_use:
            df_new[col] = df_new[col].fillna(df_new[col].mean())

    elif strategy == "Median":
        for col in cols_to_use:
            df_new[col] = df_new[col].fillna(df_new[col].median())

    elif strategy == "Mode":
        for col in cols_to_use:
            df_new[col] = df_new[col].fillna(df_new[col].mode()[0])

    elif strategy == "Drop Rows":
        df_new = df_new.dropna()

    st.session_state.df_cleaned = df_new
    st.success("Missing value strategy applied successfully ✅")

st.markdown("---")
st.subheader("📊 Before vs After Comparison")

feature = st.selectbox(
    "Select feature for comparison",
    numeric_cols
)

col1, col2 = st.columns(2)

with col1:
    st.write("Before Cleaning")
    fig_before = px.histogram(
        st.session_state.df_original,
        x=feature,
        nbins=30
    )
    st.plotly_chart(fig_before, use_container_width=True)

with col2:
    st.write("After Cleaning")
    fig_after = px.histogram(
        st.session_state.df_cleaned,
        x=feature,
        nbins=30
    )
    st.plotly_chart(fig_after, use_container_width=True)

# ================================
# STOP IF NOT TRAINED
# ================================
if not st.session_state.trained:
    st.warning("Train the model to view analysis.")
    st.stop()

model = st.session_state.model
y_pred = model.predict(X_test)

# ================================
# MAIN TABS
# ================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Visuals", "📈 Metrics", "🧠 Behaviour", "🔍 Clustering", "🤖 Recommendations", "🧼 Data Quality"]
)


# ================================
# TAB 1 — VISUALS
# ================================
with tab1:
    st.subheader("Feature Distributions")

    feature = st.selectbox(
        "Select feature",
        X.columns
    )

    fig_hist = px.histogram(
        df,
        x=feature,
        nbins=30,
        title=f"Distribution of {feature}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Outlier Detection")

    fig_box = px.box(
        df,
        y=feature,
        title=f"Outliers in {feature}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ================================
# TAB 2 — METRICS
# ================================
with tab2:
    st.subheader("Model Performance")

    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", round(acc, 3))

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("RMSE", round(rmse, 3))
        col2.metric("R²", round(r2, 3))

# ================================
# TAB 3 — BEHAVIOUR
# ================================
with tab3:
    st.subheader("Feature Importance")

    importance = None
    info_msg = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    # Linear regression (1D coef)
    elif hasattr(model, "coef_") and len(model.coef_.shape) == 1:
        importance = np.abs(model.coef_)

    # Logistic regression / multiclass (2D coef)
    elif hasattr(model, "coef_") and len(model.coef_.shape) == 2:
        importance = np.mean(np.abs(model.coef_), axis=0)
        info_msg = "ℹ Feature importance averaged across classes."

    else:
        info_msg = "ℹ Feature importance not available for this model type."

    if importance is not None:
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        if info_msg:
            st.info(info_msg)
    else:
        st.warning(info_msg)

    # ----------------------------
    # Prediction Error Analysis
    # ----------------------------
    st.subheader("Prediction Error Analysis")

    if not is_classification:
        errors = y_test - y_pred
        fig_err = px.histogram(
            errors,
            nbins=30,
            title="Residual Distribution"
        )
        st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("Residual analysis is applicable only for regression models.")

    # ----------------------------
    # SHAP Explainability
    # ----------------------------
    st.subheader("SHAP Explainability")

    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        st.write("SHAP Summary (Feature Impact)")

        fig_shap = plt.figure()

        # Handle classification vs regression safely
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values[0], X_test, show=False
            )
        else:
            shap.summary_plot(
                shap_values, X_test, show=False
            )

        st.pyplot(fig_shap)
    else:
        st.info("SHAP explainability is available only for tree-based models.")


#TAB 4 - CLUSTERING
with tab4:
    st.subheader("KMeans Clustering Behaviour")

    num_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    cluster_df = df.copy()
    cluster_df["Cluster"] = cluster_labels

    st.write("Cluster Distribution")
    st.bar_chart(cluster_df["Cluster"].value_counts())

    feature_x = st.selectbox("X-axis feature", X.columns)
    feature_y = st.selectbox("Y-axis feature", X.columns, index=1)

    fig_cluster = px.scatter(
        cluster_df,
        x=feature_x,
        y=feature_y,
        color="Cluster",
        title="Cluster Visualization"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

# ================================
# TAB 5 — LLM RECOMMENDATIONS
# ================================
with tab4:
    st.subheader("Model Improvement Suggestions")

    summary = {
        "Model": model_name,
        "Problem Type": "Classification" if is_classification else "Regression",
        "Samples": X.shape[0],
        "Features": X.shape[1]
    }

    if is_classification:
        summary["Accuracy"] = round(acc, 3)
    else:
        summary["RMSE"] = round(rmse, 3)
        summary["R2"] = round(r2, 3)

    st.json(summary)

    st.info(
        "🔮 LLM integration placeholder.\n\n"
        "You can send this summary to an LLM (OpenAI / Gemini / Ollama) "
        "to receive recommendations like:\n"
        "- Feature engineering ideas\n"
        "- Model tuning suggestions\n"
        "- Overfitting detection\n"
        "- Data quality warnings"
    )

    st.success("Dashboard analysis complete ✅")

#TAB 6
with tab6:
st.subheader("📄 Dataset Summary")

df_curr = st.session_state.df_cleaned

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
