import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="ML Model Behaviour Inspector",
    layout="wide"
    
)
st.title("ML Model Behaviour Inspector")
st.write("Analyze **how your ML model behaves**, not just how accurate it is.")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV file to get started")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# TARGET COLUMN SELECTION
# ------------------------------
st.sidebar.header("🎯 Target Column")
target_column = st.sidebar.selectbox("Select target variable", df.columns)

# ------------------------------
# BASIC PREPROCESSING
# ------------------------------
df = df.dropna()

X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical columns using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# ------------------------------
# TRAIN-TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# MODEL SELECTION
# ------------------------------
st.sidebar.header("🤖 Model Selection")

model_name = st.sidebar.selectbox(
    "Choose ML Model",
    ["Linear Regression", "Decision Tree", "Random Forest"]
)

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

# ------------------------------
# MODEL TRAINING
# ------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------
# PERFORMANCE METRICS
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("RMSE", round(rmse, 3))
col2.metric("R² Score", round(r2, 3))

# ------------------------------
# FEATURE IMPORTANCE
# ------------------------------
st.subheader("🔍 Feature Importance Analysis")

if model_name != "Linear Regression":
    importance = model.feature_importances_
else:
    importance = np.abs(model.coef_)

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# ------------------------------
# ERROR ANALYSIS
# ------------------------------
st.subheader("❌ Error Analysis")

errors = y_test - y_pred

error_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Error": errors
})

st.write("### 🔺 Top 10 Worst Predictions")
st.dataframe(error_df.reindex(error_df["Error"].abs().sort_values(ascending=False).index).head(10))

fig2, ax2 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax2)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)
st.subheader("📉 Residual Distribution")

fig3, ax3 = plt.subplots()
sns.histplot(errors, bins=30, kde=True, ax=ax3)
ax3.set_title("Residual (Error) Distribution")
st.pyplot(fig3)

st.subheader("Model Sensitivity Analysis")

selected_feature = st.selectbox("Select feature to perturb", X.columns)

delta = st.slider("Change amount (Δ)", -1.0, 1.0, 0.1)

X_modified = X_test.copy()
X_modified[selected_feature] = X_modified[selected_feature] + delta

y_modified_pred = model.predict(X_modified)

fig4, ax4 = plt.subplots()
ax4.plot(y_pred[:50], label="Original Prediction")
ax4.plot(y_modified_pred[:50], label="Modified Prediction")
ax4.set_title("Prediction Sensitivity")
ax4.legend()
st.pyplot(fig4)

st.subheader("Bias & Data Skew Detection")

skewness = df.skew(numeric_only=True)

skew_df = pd.DataFrame({
    "Feature": skewness.index,
    "Skewness": skewness.values
})

st.dataframe(skew_df)

st.warning(
    "High skewness may indicate biased learning or unstable predictions."
)
st.success("Model behaviour analysis completed successfully!")

is_classification = y.dtype == "object"
if is_classification:
    le = LabelEncoder()
    y = le.fit_transform(y)
if is_classification:
    model = RandomForestClassifier(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

st.subheader("🧼 Missing Value Analysis")

null_df = pd.DataFrame({
    "Column": df.columns,
    "Null Count": df.isnull().sum(),
    "Null %": (df.isnull().mean() * 100).round(2)
})

st.dataframe(null_df)

if null_df["Null Count"].sum() > 0:
    st.warning("Dataset contains missing values. Consider imputation.")
else:
    st.success("No missing values detected.")

import plotly.express as px

st.subheader("📊 Feature Distributions")

selected_feature = st.selectbox("Select feature", X.columns)

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f"Distribution of {selected_feature}"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("🚨 Outlier Detection")

fig_box = px.box(
    df,
    y=selected_feature,
    title=f"Outliers in {selected_feature}"
)

st.plotly_chart(fig_box, use_container_width=True)
if is_classification:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", round(acc, 3))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
else:
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.metric("RMSE", round(rmse, 3))
    st.metric("R² Score", round(r2, 3))
if is_classification:
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix"
    )

    st.plotly_chart(fig_cm, use_container_width=True)
import openai

def get_llm_recommendations(metrics_summary):
    prompt = f"""
    You are an ML expert.
    Analyze the following model performance and suggest improvements:

    {metrics_summary}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
if st.button("🤖 Get Model Improvement Suggestions"):
    summary = f"RMSE: {rmse}, R2: {r2}" if not is_classification else f"Accuracy: {acc}"
    advice = get_llm_recommendations(summary)
    st.success(advice)
