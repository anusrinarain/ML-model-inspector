import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shap
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

import google.generativeai as genai

# 1. CONFIG & STYLING 
st.set_page_config(
    page_title="ML BEHAVIOUR INSPECTOR",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fix for blank white boxes */
    div[data-testid="stMetric"] {
        background-color: #262730 !important;
        border: 1px solid #4F4F4F;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00FF00 !important;
    }
    /* Main Background adjustments */
    .main {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    /* Main Background */
    .main {
        background-color: #0E1117;
    }
    /* Card/Metric Styling */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #4F4F4F;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00FF00 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FAFAFA;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)
pio.templates.default = "plotly_dark"

# 2. LLM
@st.cache_resource
def get_gemini_response(prompt, context):
    api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        return "⚠️ Error: GEMINI_API_KEY is missing in environment variables."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        full_prompt = f"Context about the ML Model and Data:\n{context}\n\nUser Question: {prompt}\n\nAnswer as a Senior Data Scientist."
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"



# 3. SIDEBAR NAVIGATION & DATA LOADER
with st.sidebar:
    st.title("⚡ ML BEHAVIOUR INSPECTOR")
    
   

    # Navigation Menu
    selected_page = option_menu(
        menu_title=None,
        options=["Upload & Clean", "EDA & Outliers", "Model Training", "Evaluation", "AI Consultant"],
        icons=["cloud-upload", "bar-chart-line", "cpu", "check-circle", "robot"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#303030"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    
    st.markdown("---")
    
    #Data Upload 
    st.subheader("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Initialize Session State
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "model" not in st.session_state:
    st.session_state.model = None

if uploaded_file:
    if st.session_state.df_raw is None:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.session_state.df_cleaned = st.session_state.df_raw.copy()
else:
    if selected_page != "Upload & Clean":
        st.warning("Please upload a dataset in the sidebar first.")
        st.stop()


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
        
        # 3. Cleaning Pipeline
        st.markdown("### Data Cleaning Pipeline")
        col_clean1, col_clean2 = st.columns(2)
        
        with col_clean1:
            st.info("Missing Value Handling")
            
            #handle both Numbers and Text
            strategy = st.selectbox(
                "Imputation Strategy", 
                ["Drop Rows", "Auto-Clean (Mean for Num, Mode for Text)", "Fill Zero (Numeric Only)"]
            )
            
            if st.button("Apply Cleaning"):
                with st.spinner("Cleaning..."):
                    df_temp = df.copy()
                    
                    if strategy == "Drop Rows":
                        df_temp = df_temp.dropna()
                        
                    elif strategy == "Fill Zero (Numeric Only)":
                        num_cols = df_temp.select_dtypes(include=np.number).columns
                        df_temp[num_cols] = df_temp[num_cols].fillna(0)
                        
                    elif strategy == "Auto-Clean (Mean for Num, Mode for Text)":
                        # 1. Fill Numeric with MEAN
                        num_cols = df_temp.select_dtypes(include=np.number).columns
                        for col in num_cols:
                            if df_temp[col].isnull().sum() > 0:
                                df_temp[col] = df_temp[col].fillna(df_temp[col].mean())
                        
                        # 2. Fill Text/Categorical with MODE (Most Frequent)
                        cat_cols = df_temp.select_dtypes(exclude=np.number).columns
                        for col in cat_cols:
                            if df_temp[col].isnull().sum() > 0:
                                # Safe mode fill
                                mode_val = df_temp[col].mode()[0]
                                df_temp[col] = df_temp[col].fillna(mode_val)
                    
                    # Update Session State immediately
                    st.session_state.df_cleaned = df_temp
                    st.success("Cleaning Applied!")
                    st.rerun()

        with col_clean2:
            st.info("Feature Selection")
            drop_cols = st.multiselect("Select Columns to Drop", df.columns)
            if st.button("Drop Selected Columns"):
                st.session_state.df_cleaned = df.drop(columns=drop_cols)
                st.success("Columns Dropped!")
                st.rerun()

    else:
        st.info("Please upload a CSV file in the sidebar to begin.")

# PAGE 2: EDA & OUTLIERS
if selected_page == "EDA & Outliers":
    st.header("Exploratory Analysis")
    df = st.session_state.df_cleaned
    
    # Use tabs for cleaner layout
    tab1, tab2, tab3 = st.tabs(["📊Distributions & Importance", "📊Correlations", "📊Outlier Drill-Down"])
    
    with tab1:
        st.subheader("1. Feature Distribution")
        col_dist1, col_dist2 = st.columns([1, 3])
        
        with col_dist1:
            # Selector for Histogram
            feature_to_plot = st.selectbox("Select Feature to Visualize", df.columns)
            color_choice = st.color_picker("Pick a Color", "#00CC96")
        
        with col_dist2:
            #Histogram
            fig = px.histogram(
                df, 
                x=feature_to_plot, 
                marginal="box", 
                color_discrete_sequence=[color_choice], 
                title=f"Distribution: {feature_to_plot}",
                hover_data=df.columns
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Feature Importance Analysis")
        
        if st.session_state.model is not None:
            st.success(f"✅ Showing Feature Importance from trained **{st.session_state.model_type}** model.")
            
            # Extract Importances
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            
            importance_data = None
            
            if hasattr(model, "feature_importances_"):
                importance_data = model.feature_importances_
            elif hasattr(model, "coef_"):
                # Handle linear models 
                importance_data = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            
            if importance_data is not None:
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance_data
                }).sort_values("Importance", ascending=True) 
                
                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature", orientation='h', 
                    title="Global Feature Importance (From Model)",
                    color="Importance", color_continuous_scale='Bluered'
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not available for this specific model type (e.g., KNN or some SVM kernels).")
        
        else:
            
            st.warning("⚠️ No model trained yet. Showing **Correlation with Target** instead.")
            
            target_corr = st.selectbox("Select Target Variable for Correlation Check", df.columns, index=len(df.columns)-1)
            
            #Correlation
            numeric_df = df.select_dtypes(include=np.number)
            if target_corr in numeric_df.columns:
                corr_data = numeric_df.corr()[target_corr].drop(target_corr).sort_values()
                
                fig_corr_bar = px.bar(
                    x=corr_data.values, 
                    y=corr_data.index, 
                    orientation='h',
                    title=f"Correlation with '{target_corr}' (Predictive Power Proxy)",
                    labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
                    color=corr_data.values,
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr_bar, use_container_width=True)
            else:
                st.error("Selected target is not numeric. Cannot calculate correlation.")

    with tab2:
        # Correlation Matrix
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numeric columns for correlation.")

    with tab3:
        st.markdown("### Outlier Detection (IQR Method)")
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            outlier_col = st.selectbox("Select Column to Check", numeric_cols)
            
            # IQR Logic
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Outliers", len(outliers))
            c2.metric("Lower Limit", f"{lower_bound:.2f}")
            c3.metric("Upper Limit", f"{upper_bound:.2f}")
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_box = px.box(df, y=outlier_col, title=f"Box Plot: {outlier_col}", color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col_g2:
                fig_out = px.scatter(df, y=outlier_col, title=f"Scatter Spread: {outlier_col}", color_discrete_sequence=['#636EFA'])
                fig_out.add_hrect(y0=lower_bound, y1=upper_bound, line_width=0, fillcolor="green", opacity=0.1)
                st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.warning("No numeric columns found for outlier detection.")

# PAGE 3: MODEL TRAINING 
if selected_page == "Model Training":
    st.header("Model Training")
    
    df = st.session_state.df_cleaned.copy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Target & Model")
    
    target_col = st.sidebar.selectbox("Select Target Variable (y)", df.columns)
    problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])
    
    # Preprocessing
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = pd.get_dummies(X, drop_first=True)
    if problem_type == "Classification" and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    if problem_type == "Classification":
        model_name = st.sidebar.selectbox("Choose Algorithm", 
            ["Logistic Regression", "Random Forest", "SVC (SVM)", "Gradient Boosting", "Neural Network (MLP)"])
    else:
        model_name = st.sidebar.selectbox("Choose Algorithm", 
            ["Linear Regression", "Random Forest", "SVR (SVM)", "Gradient Boosting", "Neural Network (MLP)"])
            
    st.sidebar.subheader("Validation")
    val_method = st.sidebar.radio("Validation Method", ["Train-Test Split", "K-Fold Cross Validation"])
    
    test_size = 0.2
    k_folds = 5
    
    if val_method == "Train-Test Split":
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    else:
        k_folds = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)
        st.sidebar.info("ℹ️ Use K-Fold when dataset is small (<1000 rows) to get reliable accuracy.")

    st.subheader(f"Current Config: {model_name} ({problem_type})")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("Input Features (X):", X.shape)
    with col_t2:
        st.write("Target (y):", target_col)

    if st.button("Start Training", type="primary"):
        with st.spinner("Training Model..."):
            
            # Scale Data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Instantiate Model
            if problem_type == "Classification":
                if model_name == "Logistic Regression": model = LogisticRegression()
                elif model_name == "Random Forest": model = RandomForestClassifier()
                elif model_name == "SVC (SVM)": model = SVC(probability=True)
                elif model_name == "Gradient Boosting": model = GradientBoostingClassifier()
                elif model_name == "Neural Network (MLP)": model = MLPClassifier(max_iter=500)
            else: # Regression
                if model_name == "Linear Regression": model = LinearRegression()
                elif model_name == "Random Forest": model = RandomForestRegressor()
                elif model_name == "SVR (SVM)": model = SVR()
                elif model_name == "Gradient Boosting": model = GradientBoostingRegressor()
                elif model_name == "Neural Network (MLP)": model = MLPRegressor(max_iter=500)
            
            # Train Logic
            if val_method == "Train-Test Split":
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train 
            else:
                # K-Fold Logic
                scores = cross_val_score(model, X_scaled, y, cv=k_folds)
                score = scores.mean()
                # full data for final model
                model.fit(X_scaled, y)
                st.session_state.X_test = X_scaled 
                st.session_state.y_test = y
                st.session_state.X_train = X_scaled

            st.session_state.model = model
            st.session_state.model_type = problem_type
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.train_score = score
            st.session_state.target_name = target_col
            
            st.success(f"Training Complete! Score: {score:.4f}")
            st.balloons()


# PAGE 4: EVALUATION
if selected_page == "Evaluation":
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
        st.stop()
        
    st.header("📈 Performance Dashboard")
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    y_pred = model.predict(X_test)
    
    # 1. Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    if st.session_state.model_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        c1.metric("Accuracy", f"{acc:.2%}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", title="Heatmap")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            st.subheader("Prediction Error Distribution")
            # DataFrame for Actual vs Predicted
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            count_df = results_df.groupby(["Actual", "Predicted"]).size().reset_index(name="Count")
            
            # Bar Chart
            fig_err = px.bar(
                count_df, 
                x="Actual", 
                y="Count", 
                color="Predicted", 
                title="Actual vs Predicted Counts",
                barmode="group"
            )
            st.plotly_chart(fig_err, use_container_width=True)
            
            # 1. Create the Heatmap
            fig_cm = px.imshow(
                cm, 
                text_auto=True, 
                color_continuous_scale="Viridis",
                labels=dict(x="Predicted Class", y="Actual Class"),
                x=[f"Class {i}" for i in range(len(cm))],
                y=[f"Class {i}" for i in range(len(cm))]
            )
            
            fig_cm.update_layout(
                height=600,  
                width=800,   
                dragmode='zoom', 
                font=dict(size=14)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                st.subheader("ROC Curve")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.area(
                    x=fpr, y=tpr, 
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                    color_discrete_sequence=['#AB63FA']
                )
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC Curve is available for Binary Classification models that support probabilities.")

    else: # Regression
        r2 = r2_score(y_test, y_pred)
        c1.metric("R2 Score", f"{r2:.3f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
        
        c_p1, c_p2 = st.columns(2)
        
        with c_p1:
            st.subheader("Actual vs Predicted")
            df_res = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig_res = px.scatter(df_res, x="Actual", y="Predicted", trendline="ols", title="Prediction Alignment")
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c_p2:
            st.subheader("Residual Distribution")
            residuals = y_test - y_pred
            fig_hist_res = px.histogram(residuals, nbins=30, title="Error Distribution", color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig_hist_res, use_container_width=True)

# PAGE 5: AI CONSULTANT
if selected_page == "AI Consultant":
    st.header("AI Assistant")
  

    # 2. Context Builder
    if st.session_state.model is not None:
        context = f"""
        Model Type: {st.session_state.model_type}
        Target Variable: {st.session_state.target_name}
        Performance Score: {st.session_state.train_score:.3f}
        Features Used: {st.session_state.feature_names}
        """
        st.success("Context Loaded from Trained Model")
    else:
        context = "User has not trained a model yet. Guide them on general ML best practices."
        st.warning("⚠️ No model trained yet. AI responses will be generic.")

    # 3. Chat Interface
    user_query = st.chat_input("Ask about your model (e.g., 'Why is my accuracy low?' or 'Explain the confusion matrix')")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display Chat
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)
            
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response(user_query, context)
                st.write(response)
                st.session_state.chat_history.append(("assistant", response))