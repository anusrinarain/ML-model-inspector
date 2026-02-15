import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import os
import random
import time

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

import google.generativeai as genai
#1. CONFIG & STYLING 
st.set_page_config(
    page_title="InspectorML Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #262730 !important;
        border: 1px solid #4F4F4F;
        color: white;
    }
    div[data-testid="stMetricLabel"] { color: #B0B0B0 !important; }
    div[data-testid="stMetricValue"] { color: #00FF00 !important; }
    .main { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)
pio.templates.default = "plotly_dark"
# 2. LLM 
@st.cache_resource
def get_gemini_response(user_query, context):
    api_key = None
    try: api_key = st.secrets["GEMINI_API_KEY"]
    except: api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ Note: API Key missing. Please check .streamlit/secrets.toml"

    try:
        genai.configure(api_key=api_key)
        valid_model = "gemini-1.5-flash"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name or 'pro' in m.name:
                        valid_model = m.name
                        break
        except: pass

        model = genai.GenerativeModel(valid_model)
        full_prompt = f"""
        ROLE: Act as a Senior Data Scientist and Machine Learning Engineer.
        TASK: Evaluate the following context and answer the user's question in detail.
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {user_query}
        
        GUIDELINES:
        - Be professional, concise, and insightful.
        - Explain *why* something is happening, not just *what*.
        - If suggesting fixes, give specific examples.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"🤖 AI Service Error: {str(e)}"
# 3. SIDEBA
with st.sidebar:
    st.title("⚡ InspectorML Pro")
    selected_page = option_menu(
        menu_title=None,
        options=["Upload & Clean", "EDA & Outliers", "Model Training", "Evaluation", "Auto-Governance", "Risk & Simulation Lab", "AI Consultant"],
        icons=["cloud-upload", "bar-chart-line", "cpu", "check-circle", "trophy", "virus", "robot"],
        default_index=0,
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if "df_cleaned" not in st.session_state: st.session_state.df_cleaned = None
if "model" not in st.session_state: st.session_state.model = None

if uploaded_file:
    if st.session_state.df_cleaned is None:
        st.session_state.df_cleaned = pd.read_csv(uploaded_file)

# PAGE 1: UPLOAD & CLEAN
if selected_page == "Upload & Clean":
    st.header("Data Setup Center")
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=300) 
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Quick Cleaning")
            if st.button("Auto-Clean (Mean/Mode)"):
                df_temp = df.copy()
                for c in df_temp.columns:
                    if df_temp[c].dtype == 'object': df_temp[c] = df_temp[c].fillna(df_temp[c].mode()[0])
                    else: df_temp[c] = df_temp[c].fillna(df_temp[c].mean())
                st.session_state.df_cleaned = df_temp
                st.success("Dataset Cleaned!")
                st.rerun()
        with col2:
            st.info("Drop Columns")
            drop_cols = st.multiselect("Select Columns", df.columns)
            if st.button("Drop Selected"):
                st.session_state.df_cleaned = df.drop(columns=drop_cols)
                st.rerun()
    else:
        st.info("Please upload a CSV file.")


# PAGE 2: EDA 
if selected_page == "EDA & Outliers":
    st.header("Exploratory Analysis")
    df = st.session_state.df_cleaned
    
    tab1, tab2, tab3 = st.tabs(["Distributions & Importance", "Correlations", "Outliers (Detailed)"])
    
    with tab1:
        c1, c2 = st.columns([1,3])
        with c1: feat = st.selectbox("Select Feature", df.columns)
        with c2: st.plotly_chart(px.histogram(df, x=feat, marginal="box", color_discrete_sequence=['#00CC96'], title=f"Distribution of {feat}"), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Importance")
        if st.session_state.model is not None:
            model = st.session_state.model
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({"Feature": st.session_state.feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation='h', title="Model Feature Importance", color="Importance"), use_container_width=True)
            elif hasattr(model, "coef_"):
                imp = pd.DataFrame({"Feature": st.session_state.feature_names, "Importance": np.abs(model.coef_[0]) if len(model.coef_.shape)>1 else np.abs(model.coef_)}).sort_values("Importance", ascending=True)
                st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation='h', title="Model Coefficients", color="Importance"), use_container_width=True)
            else:
                st.info("Selected model does not provide feature importance. (Showing Correlation below instead)")
        if st.session_state.get("target_name") in df.columns:
            target = st.session_state.target_name
            num_df = df.select_dtypes(include=np.number)
            if target in num_df.columns:
                corr = num_df.corr()[target].drop(target).sort_values()
                st.plotly_chart(px.bar(x=corr.values, y=corr.index, orientation='h', title=f"Correlation with Target ({target})", labels={'x':'Correlation', 'y':'Feature'}), use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            st.plotly_chart(px.imshow(num_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap"), use_container_width=True)

    with tab3:
        st.subheader("Outlier Inspector")
        num_cols = df.select_dtypes(include=np.number).columns
        col_out = st.selectbox("Select Column to Inspect", num_cols)
        v1, v2 = st.tabs(["Box Plot View", "Scatter View"])
        
        with v1:
            st.plotly_chart(px.box(df, y=col_out, points="all", title=f"Box Plot: {col_out}", color_discrete_sequence=['#FF4B4B']), use_container_width=True)
        with v2:
            st.plotly_chart(px.scatter(df, y=col_out, title=f"Scatter Distribution: {col_out}", color_discrete_sequence=['#636EFA']), use_container_width=True)

# PAGE 3: TRAINING 
if selected_page == "Model Training":
    st.header("Model Training")
    df = st.session_state.df_cleaned.copy()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    target_col = st.sidebar.selectbox("Select Target Variable", df.columns)
    
    
    target_type = df[target_col].dtype
    st.sidebar.info(f"Target Type: **{target_type}**")
    if pd.api.types.is_numeric_dtype(target_type) and df[target_col].nunique() > 10:
        rec_type = "Regression"
    else:
        rec_type = "Classification"
    st.sidebar.caption(f"Recommended: {rec_type}")

    problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"], index=0 if rec_type=="Classification" else 1)
    
    # Model Selection
    if problem_type == "Classification":
        model_name = st.sidebar.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (SVC)", "Neural Network (MLP)"])
    else:
        model_name = st.sidebar.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Gradient Boosting", "SVM (SVR)", "Neural Network (MLP)"])
            
    val_method = st.sidebar.radio("Validation", ["Train-Test Split", "K-Fold"])
    
    # RESTORED: Config Display
    st.subheader("⚙️ Model Configuration")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Target:** {target_col}")
    c2.info(f"**Type:** {problem_type}")
    c3.info(f"**Model:** {model_name}")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training in progress..."):
            X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
            y = df[target_col]
            if problem_type == "Classification" and y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.scaler = scaler
            
            # Models
            if problem_type == "Classification":
                if "Logistic" in model_name: model = LogisticRegression(max_iter=1000)
                elif "Random Forest" in model_name: model = RandomForestClassifier()
                elif "Gradient" in model_name: model = GradientBoostingClassifier()
                elif "SVM" in model_name: model = SVC(probability=True)
                elif "Neural" in model_name: model = MLPClassifier(max_iter=500)
            else:
                if "Linear" in model_name: model = LinearRegression()
                elif "Random Forest" in model_name: model = RandomForestRegressor()
                elif "Gradient" in model_name: model = GradientBoostingRegressor()
                elif "SVM" in model_name: model = SVR()
                elif "Neural" in model_name: model = MLPRegressor(max_iter=500)
            
            # Training
            if val_method == "Train-Test Split":
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
            else:
                scores = cross_val_score(model, X_scaled, y, cv=5)
                score = scores.mean()
                model.fit(X_scaled, y)
                st.session_state.X_test = X_scaled
                st.session_state.y_test = y

            st.session_state.model = model
            st.session_state.model_type = problem_type
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.train_score = score
            st.session_state.target_name = target_col
            
            st.success(f"Training Complete! Score: {score:.4f}")
            st.balloons() 


# PAGE 4: EVALUATION 
if selected_page == "Evaluation":
    if st.session_state.model is None: st.warning("Train first."); st.stop()
    st.header("Performance Dashboard")
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = model.predict(X_test)
    
    # 1. METRICS
    c1, c2, c3, c4 = st.columns(4)
    if st.session_state.model_type == "Classification":
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        
        # 2. GRAPHS
        tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])
        with tab1:
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix"), use_container_width=True)
        with tab2:
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                st.plotly_chart(px.area(x=fpr, y=tpr, title="ROC Curve", labels={'x':'FPR', 'y':'TPR'}), use_container_width=True)
            else: st.info("ROC available for Binary Classification only.")
            
    else: # Regression
        c1.metric("R2 Score", f"{r2_score(y_test, y_pred):.3f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        c3.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
        c4.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
        
        # 2. GRAPHS (RESTORED ACTUAL VS PREDICTED)
        tab1, tab2 = st.tabs(["Actual vs Predicted", "Residuals"])
        with tab1:
            # The exact graph you liked
            df_res = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.plotly_chart(px.scatter(df_res, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted Correlation", color_discrete_sequence=['#00CC96']), use_container_width=True)
        with tab2:
            residuals = y_test - y_pred
            st.plotly_chart(px.histogram(residuals, nbins=30, title="Residual Error Distribution", color_discrete_sequence=['#FF4B4B']), use_container_width=True)


# PAGE 5: AUTO-GOVERNANCE
if selected_page == "Auto-Governance":
    st.header("Automated Model Tournament")
    if st.session_state.df_cleaned is None: st.stop()
    
    target = st.selectbox("Target", st.session_state.df_cleaned.columns)
    prob_type = st.radio("Type", ["Classification", "Regression"])
    
    if st.button("Run Tournament"):
        with st.spinner("Running Tournament..."):
            try:
                X = pd.get_dummies(st.session_state.df_cleaned.drop(columns=[target]), drop_first=True)
                y = st.session_state.df_cleaned[target]
                if prob_type == "Classification" and y.dtype == 'object': y = LabelEncoder().fit_transform(y)
                X_sc = StandardScaler().fit_transform(X)
                
                if prob_type == "Classification":
                    models = {"RandomForest": RandomForestClassifier(), "Logistic": LogisticRegression(), "SVM": SVC(), "MLP": MLPClassifier(max_iter=300)}
                else:
                    models = {"RandomForest": RandomForestRegressor(), "Linear": LinearRegression(), "SVM": SVR(), "MLP": MLPRegressor(max_iter=300)}
                
                results = []
                cv = 3 if len(y) > 50 else 2 # Safety check
                
                for name, m in models.items():
                    try:
                        scores = cross_val_score(m, X_sc, y, cv=cv)
                        results.append({"Model": name, "Score": scores.mean()})
                    except: pass
                
                res_df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
                st.session_state.gov_results = res_df
                st.session_state.best_gov_model = models[res_df.iloc[0]["Model"]] 
                
            except Exception as e: st.error(f"Error: {e}")
    
    if "gov_results" in st.session_state:
        st.dataframe(st.session_state.gov_results.style.highlight_max(subset="Score", color="#00CC96"), use_container_width=True)
        
        if st.button("Activate Best Model"):
            st.session_state.model = st.session_state.best_gov_model
            # Re-fit
            X = pd.get_dummies(st.session_state.df_cleaned.drop(columns=[target]), drop_first=True)
            y = st.session_state.df_cleaned[target]
            if prob_type == "Classification" and y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            st.session_state.model.fit(StandardScaler().fit_transform(X), y)
            st.success(f"Activated {st.session_state.gov_results.iloc[0]['Model']}!")


# PAGE 6: SIMULATION 
if selected_page == "Risk & Simulation Lab":
    st.header("Simulation Lab")
    if st.session_state.model is None: st.warning("Train first."); st.stop()
    
    df = st.session_state.df_cleaned
    target = st.session_state.target_name
    X_str = df.drop(columns=[target])
    c1, c2 = st.columns(2)
    if c1.button("Force Safe Values"):
        st.session_state.force_safe = True
        st.session_state.force_risk = False
        st.rerun()
    if c2.button("Force Risk Values"):
        st.session_state.force_safe = False
        st.session_state.force_risk = True
        st.rerun()

    inputs = {}
    cols = st.columns(3)
    
    for i, col in enumerate(X_str.columns):
        with cols[i%3]:
            if pd.api.types.is_numeric_dtype(X_str[col]):
                mn, mx = float(X_str[col].min()), float(X_str[col].max())
                avg = float(X_str[col].mean())
                val = avg
                if st.session_state.get("force_safe"): val = avg
                if st.session_state.get("force_risk"): val = mx if i % 2 == 0 else mn # Chaos
                
                inputs[col] = st.slider(col, mn, mx, val, key=f"sl_{col}")
            else:
                inputs[col] = st.selectbox(col, X_str[col].unique())
    
    if st.button("Simulate Outcome", type="primary"):
        # Prediction
        input_enc = pd.get_dummies(pd.DataFrame([inputs]))
        for c in st.session_state.feature_names:
            if c not in input_enc.columns: input_enc[c] = 0
        final_in = st.session_state.scaler.transform(input_enc[st.session_state.feature_names])
        pred = st.session_state.model.predict(final_in)[0]
        
        # Risk Logic
        prob = 0
        if hasattr(st.session_state.model, "predict_proba"):
            probs = st.session_state.model.predict_proba(final_in)[0]
            prob = probs[1] if len(probs) == 2 else np.max(probs)
            
        # Display
        st.subheader("Simulation Result")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Output", str(pred))
        c2.metric("Risk Probability", f"{prob:.1%}")
        
        # CLEAN PROBABILITY BAR (Replaces confusing gauge)
        st.progress(prob)
        if prob > 0.7: st.error("HIGH RISK DETECTED")
        elif prob > 0.4: st.warning("MODERATE RISK")
        else: st.success("LOW RISK / SAFE")
        
        # GEMINI EXPLANATION BUTTON
        if st.button("Ask AI to Explain This Result"):
            with st.spinner("AI Analyzing..."):
                prompt = f"The model predicted {pred} with {prob:.1%} probability for inputs: {inputs}. Explain why."
                st.write(get_gemini_response(prompt, "Simulation Context"))


# PAGE 7: AI CONSULTANt
if selected_page == "AI Consultant":
    st.header("AI Assistant")
    
    if st.session_state.model is not None:
        context = f"Model: {st.session_state.model_type}, Target: {st.session_state.target_name}, Score: {st.session_state.train_score:.3f}"
        st.success("Context Loaded")
    
    q = st.chat_input("Ask about your model...")
    if q:
        st.write(f"**You:** {q}")
        with st.spinner("AI Thinking..."):
            st.write(get_gemini_response(q, context))