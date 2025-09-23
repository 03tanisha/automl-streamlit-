import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from models import (
    linear_regression_train,
    logistic_regression_train,
    lasso_train,
    ridge_train,
    decision_tree_classifier_train,
    decision_tree_regressor_train,
    random_forest_classifier_train,
    random_forest_regressor_train,
    xgboost_train
)

# Inject CSS directly in Streamlit
st.markdown(
    """
    <style>

    /* Add spacing above all Streamlit dataframes */
    [data-testid="stDataFrame"] {
        margin-top: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Page config -------------------
st.set_page_config(page_title="AutoML", layout="wide")

# ------------------- Initialize session state -------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Overview"

if "df" not in st.session_state:
    st.session_state.df = None

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None


# ------------------- Load CSS -------------------
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# ------------------- Heading -------------------
st.markdown('<h1 class="center-heading">AutoML</h1>', unsafe_allow_html=True)
# --- Tabs ---
tabs = ["📊 Overview", "📋 Data Preview", "📈 Visualizations", "🤖 Model Training", "🔍 Testing"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Overview"

selected_tab = st.radio(
    "",
    options=tabs,
    horizontal=True,
    index=tabs.index(st.session_state.active_tab),
    label_visibility="collapsed"
)

# --- Update session state ---
st.session_state.active_tab = selected_tab

# --- Line below tabs ---
st.markdown('<div class="heading-tabsep"></div>', unsafe_allow_html=True)







# ------------------- Overview Tab -------------------
if st.session_state.active_tab == "📊 Overview":
    st.markdown('<p class="intro-text-short">Turn your data into insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="intro-text">Build ML models in minutes — upload, explore, and train effortlessly</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="get-started tab-separator">Getting Started</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-box">
        <span>📂 Upload your CSV file</span>
        <span>👀 Preview your data</span>
        <span>📊 Generate Visualizations</span>
        <span>⚙️ Select algorithms and tune parameters</span>
        <span>🧪 Testing</span>
        <span>💡 Get insights</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("File requirements"):
        st.markdown("""
        - File format: **CSV**
        - Maximum file size: **100MB**
        - Supported Columns:
            - **Numerical** → (12, 3.5, 11)
            - **Categorical** → (cat, dog, true/false)
            - **Temporal** → (2024-08-28)
        """)

    with st.expander("Example datasets"):
        st.markdown("""
        Try these example datasets:
        - [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)
        - [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
        - [Credit Card Dataset](https://archive.ics.uci.edu/dataset/27/credit+approval)
        """)

    # st.markdown('div class="upload-file tab-separator">📤 Upload Your Dataset")
    st.markdown('<div class="upload-file tab-separator">📤 Upload Your Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop your CSV file here or click to browse", type="csv")
    skip_cleaning = st.checkbox("My dataset is already cleaned (skip data cleaning)")

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        # st.markdown('<div class="custom-progress">', unsafe_allow_html=True)
        progress_text = st.empty()
        progress_bar = st.progress(0)
        # st.markdown('</div>', unsafe_allow_html=True)
        # for percent_complete in range(100):
        #     progress_text.text(f"Processing... {percent_complete + 1}%")
        #     progress_bar.progress(percent_complete + 1)
        #     time.sleep(0.01)
        for percent_complete in range(101):
    # progress bar itself
            progress_bar.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {percent_complete}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # progress text
            progress_text.markdown(f'<div class="progress-text">Processing... {percent_complete}%</div>', unsafe_allow_html=True)
            
            time.sleep(0.01)

        st.session_state.df = pd.read_csv(uploaded_file)
        st.markdown('<div class="file-upload">File uploaded and cleaned successfully', unsafe_allow_html=True)


# ------------------- Data Preview Tab -------------------
elif st.session_state.active_tab == "📋 Data Preview":
    if st.session_state.df is not None:
        df = st.session_state.df


        st.markdown('<div class="shape tab-separator">🎯 Shape of Data</div>', unsafe_allow_html=True)
        st.markdown(
                    f"""
                    <ul class="shape">
                        <li><span class="label">Total number of rows:</span> <span class="value">{df.shape[0]}</span></li>
                        <li><span class="label">Total number of columns:</span> <span class="value">{df.shape[1]}</span></li>
                    </ul>
                    """,
                    unsafe_allow_html=True
                )
        

        st.markdown('<div class="col-detect-heading tab-separator">🔍 Columns Detected</div>', unsafe_allow_html=True)

        cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

            # Sections (Django-style)
        if cat_cols:
            st.markdown(
                f"""
                <div class="column-section">
                    <div class="column-header">
                        <span class="column-label">Categorical Columns:</span> There are total {len(cat_cols)} columns
                    </div>
                    <div class="column-names">{", ".join(cat_cols)}</div>
                </div>
                """, unsafe_allow_html=True
            )

        if num_cols:
            st.markdown(
                f"""
                <div class="column-section">
                    <div class="column-header">
                        <span class="column-label">Numerical Columns:</span> There are total {len(num_cols)} columns
                    </div>
                    <div class="column-names">{", ".join(num_cols)}</div>
                </div>
                """, unsafe_allow_html=True
            )

        if date_cols:
            st.markdown(
                f"""
                <div class="column-section">
                    <div class="column-header">
                        <span class="column-label">Datetime Columns:</span> There are total {len(date_cols)} columns
                    </div>
                    <div class="column-names">{", ".join(date_cols)}</div>
                </div>
                """, unsafe_allow_html=True
            )

        # st.markdown('div class="summary-heading tab-separator">💹 Summary Stats</div>', unsafe_allow_html=True)
        st.markdown('<div class="shape tab-separator">💹 Summary Stats</div>', unsafe_allow_html=True)
        st.dataframe(df.describe())

        # st.markdown('div class="col-info tab-separator">📌 Column Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="shape tab-separator">📌 Column Information</div>', unsafe_allow_html=True)

        col_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes,
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(col_info)
    else:
        st.info("Upload a CSV file in the Overview tab to see the data preview.")

# ------------------- Visualizations Tab -------------------
elif st.session_state.active_tab == "📈 Visualizations":
    if st.session_state.df is not None:
        df = st.session_state.df
        cols = df.columns.tolist()
        visuals = st.multiselect("Select columns to visualize", options=cols, default=None)

        # --- Dark theme function ---
        def apply_dark_theme(fig, grid=True, height=250):
            fig.update_layout(
                plot_bgcolor="#222",
                paper_bgcolor="#222",
                font=dict(color="white"),
                xaxis=dict(showgrid=grid, gridcolor="#444"),
                yaxis=dict(showgrid=grid, gridcolor="#444"),
                height=height,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            return fig

        # --- Chart generation ---
        def generate_charts(df, col):
            charts = []
            if pd.api.types.is_numeric_dtype(df[col]):
                # Histogram
                fig_hist = px.histogram(df, x=col, nbins=18, title=f"Histogram of {col}")
                fig_hist.update_traces(marker_color="#58508d", hovertemplate="%{x}:%{y}")
                charts.append(apply_dark_theme(fig_hist, height=250))

                # Boxplot
                fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
                fig_box.update_traces(marker_color="#bc5090", hovertemplate="%{y}")
                charts.append(apply_dark_theme(fig_box, grid=False, height=250))
            else:
                # Top 5 categories for bar chart
                top_values = df[col].value_counts().nlargest(5)
                
                # Bar chart
                fig_bar = px.bar(x=top_values.index, y=top_values.values, title=f"Top 5 values in {col}")
                fig_bar.update_traces(marker_color="#58508d", hovertemplate="%{x}: %{y}")
                charts.append(apply_dark_theme(fig_bar, height=250))

                # Pie chart
                fig_pie = px.pie(values=top_values.values, names=top_values.index, title=f"Pie chart of {col}")
                fig_pie.update_traces(hoverinfo='label+percent', textinfo='value')
                charts.append(apply_dark_theme(fig_pie, grid=False, height=250))
            return charts

        if visuals:
            for i in range(0, len(visuals), 2):
                row_cols = st.columns(4)
                for j in range(2):
                    if i + j < len(visuals):
                        col_name = visuals[i + j]
                        charts = generate_charts(df, col_name)
                        row_cols[j*2].plotly_chart(charts[0], use_container_width=True)
                        row_cols[j*2 + 1].plotly_chart(charts[1], use_container_width=True)

        # st.markdown('<div class="upload-file tab-separator">Correlation Matrix</div>', unsafe_allow_html=True)
        # num_cols = df.select_dtypes(include=['int64', 'float64'])
        # if not num_cols.empty:
        #     corr = num_cols.corr()
        #     fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
        #                     aspect='auto', title="Correlation Heatmap")
        #     fig.update_layout(margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor='rgba(0,0,0,1)',
        #                       plot_bgcolor='rgba(0,0,0,1)', font_color='white', height=250,width=150)
        #     st.plotly_chart(fig)
        # else:
        #     st.warning("⚠️ No numeric columns available for the correlation matrix")
        st.markdown('<div class="upload-file tab-separator">Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=['int64', 'float64'])
        if not num_cols.empty:
            corr = num_cols.corr()
            fig = px.imshow(
                corr, text_auto=True, color_continuous_scale='Viridis', aspect='auto', title="Correlation Heatmap"
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,1)',
                plot_bgcolor='rgba(0,0,0,1)',
                font_color='white',
                height=300   # reduced height
            )
            
            # Wrap in a container to control width
            st.markdown('<div style="max-width:100px; margin:auto;">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No numeric columns available for the correlation matrix")
        
    else:
        st.info("Upload a CSV file in the Overview tab to generate visualizations.")

# ------------------- Model Training Tab -------------------
# ------------------- Model Training Tab -------------------
elif st.session_state.active_tab == "🤖 Model Training":
    if st.session_state.df is None:
        st.warning("Please upload the file first!")
    else:
        df = st.session_state.df
        # st.subheader("Train Model")
        models = [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Logistic Regression",
            "Decision Tree Classifier",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Random Forest Classifier",
            "XGBoost"
        ]
        selected_model = st.selectbox("Select model to train:", options=models)
        target_col = st.selectbox("Select target column(y)", options=df.columns.tolist())
        problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target_col]) else "Classification"
        # st.info(f"Problem type detected: {problem_type}")
        # st.markdown('<div class="shape tab-separator">📌 Column Information</div>', unsafe_allow_html=True)
        # st.markdown('<div class="prob-type">Problem type detected:{problem_type}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="prob-type">Problem type detected: {problem_type}</div>', unsafe_allow_html=True)


        # --- Hyperparameters ---
        st.markdown('<div class="hyper-params tab-separator">Hyperparameter</div>', unsafe_allow_html=True)
        hyperparams = {}
        if selected_model == "Linear Regression":
            st.markdown('<div class="linear-para">No hyperparameters available for Linear Regression</div>', unsafe_allow_html=True)
        elif selected_model == "Lasso Regression":
            alpha = st.number_input("Alpha (regularization strength)", value=1.0, step=0.1)
            hyperparams["alpha"] = float(alpha)
        elif selected_model == "Ridge Regression":
            alpha = st.number_input("Alpha (regularization strength)", value=1.0, step=0.1)
            hyperparams["alpha"] = float(alpha)
        elif selected_model == "Logistic Regression":
            C = st.number_input("Regularization strength (C)", value=1.0, step=0.1)
            penalty = st.selectbox("Penalty", options=["l1", "l2"])
            hyperparams["C"] = float(C)
            hyperparams["penalty"] = penalty
        elif selected_model in ["Decision Tree Classifier", "Decision Tree Regressor"]:
            criterion_options = ["gini", "entropy"] if "Classifier" in selected_model else ["squared_error", "friedman_mse", "absolute_error"]
            criterion = st.selectbox("Criterion", options=criterion_options)
            max_depth = st.number_input("Max depth", min_value=1, value=5)
            min_samples_split = st.number_input("Min samples split", min_value=2, value=2)
            hyperparams["criterion"] = criterion
            hyperparams["max_depth"] = int(max_depth)
            hyperparams["min_samples_split"] = int(min_samples_split)
        elif selected_model in ["Random Forest Regressor", "Random Forest Classifier"]:
            n_estimators = st.number_input("Number of trees", min_value=10, value=100)
            max_depth = st.number_input("Max depth", min_value=1, value=5)
            min_samples_split = st.number_input("Min samples split", min_value=2, value=2)
            hyperparams["n_estimators"] = int(n_estimators)
            hyperparams["max_depth"] = int(max_depth)
            hyperparams["min_samples_split"] = int(min_samples_split)
        elif selected_model == "XGBoost":
            n_estimators = st.number_input("Number of trees", min_value=10, value=100)
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
            max_depth = st.number_input("Max depth", min_value=1, value=3)
            hyperparams["n_estimators"] = int(n_estimators)
            hyperparams["learning_rate"] = float(learning_rate)
            hyperparams["max_depth"] = int(max_depth)

        if st.button("Train Model"):
            try:
                if selected_model == "Linear Regression":
                    pipe, X_test, y_test = linear_regression_train(df, target_col)
                elif selected_model == "Lasso Regression":
                    pipe, X_test, y_test = lasso_train(df, target_col, **hyperparams)
                elif selected_model == "Ridge Regression":
                    pipe, X_test, y_test = ridge_train(df, target_col, **hyperparams)
                elif selected_model == "Logistic Regression":
                    pipe, X_test, y_test = logistic_regression_train(df, target_col, **hyperparams)
                elif selected_model == "Decision Tree Classifier":
                    pipe, X_test, y_test = decision_tree_classifier_train(df, target_col, **hyperparams)
                elif selected_model == "Decision Tree Regressor":
                    pipe, X_test, y_test = decision_tree_regressor_train(df, target_col, **hyperparams)
                elif selected_model == "Random Forest Classifier":
                    pipe, X_test, y_test = random_forest_classifier_train(df, target_col, **hyperparams)
                elif selected_model == "Random Forest Regressor":
                    pipe, X_test, y_test = random_forest_regressor_train(df, target_col, **hyperparams)
                elif selected_model == "XGBoost":
                    pipe, X_test, y_test = xgboost_train(df, target_col, **hyperparams)
                else:
                    st.warning("Selected model not implemented yet")
                    pipe, X_test, y_test = None, None, None

                if pipe is not None:
                    st.markdown(f'<div class="train-model">Trained {selected_model} successfully!</div>', unsafe_allow_html=True)

                    st.session_state['trained_model'] = {"pipe": pipe, "X_test": X_test, "y_test": y_test}
            except Exception as e:
                st.error(f"Error training {selected_model}: {e}")

# elif st.session_state.active_tab == "🤖 Model Training":
#     if st.session_state.df is not None:
#         df = st.session_state.df
#         # st.subheader("Train Model")
#         models = [
#             "Linear Regression",
#             "Lasso Regression",
#             "Ridge Regression",
#             "Logistic Regression",
#             "Decision Tree Classifier",
#             "Decision Tree Regressor",
#             "Random Forest Regressor",
#             "Random Forest Classifier",
#             "XGBoost"
#         ]
#         selected_model = st.selectbox("Select model to train:", options=models)
#         target_col = st.selectbox("Select target column(y)", options=df.columns.tolist())
#         problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target_col]) else "Classification"
#         # st.info(f"Problem type detected: {problem_type}")
#         # st.markdown('<div class="shape tab-separator">📌 Column Information</div>', unsafe_allow_html=True)
#         # st.markdown('<div class="prob-type">Problem type detected:{problem_type}</div>',unsafe_allow_html=True)
#         st.markdown(f'<div class="prob-type">Problem type detected: {problem_type}</div>', unsafe_allow_html=True)


#         # --- Hyperparameters ---
#         st.markdown('<div class="hyper-params tab-separator">Hyperparameter</div>', unsafe_allow_html=True)
#         hyperparams = {}
#         if selected_model == "Linear Regression":
#             st.markdown('<div class="linear-para">No hyperparameters available for Linear Regression</div>', unsafe_allow_html=True)
#         elif selected_model == "Lasso Regression":
#             alpha = st.number_input("Alpha (regularization strength)", value=1.0, step=0.1)
#             hyperparams["alpha"] = float(alpha)
#         elif selected_model == "Ridge Regression":
#             alpha = st.number_input("Alpha (regularization strength)", value=1.0, step=0.1)
#             hyperparams["alpha"] = float(alpha)
#         elif selected_model == "Logistic Regression":
#             C = st.number_input("Regularization strength (C)", value=1.0, step=0.1)
#             penalty = st.selectbox("Penalty", options=["l1", "l2"])
#             hyperparams["C"] = float(C)
#             hyperparams["penalty"] = penalty
#         elif selected_model in ["Decision Tree Classifier", "Decision Tree Regressor"]:
#             criterion_options = ["gini", "entropy"] if "Classifier" in selected_model else ["squared_error", "friedman_mse", "absolute_error"]
#             criterion = st.selectbox("Criterion", options=criterion_options)
#             max_depth = st.number_input("Max depth", min_value=1, value=5)
#             min_samples_split = st.number_input("Min samples split", min_value=2, value=2)
#             hyperparams["criterion"] = criterion
#             hyperparams["max_depth"] = int(max_depth)
#             hyperparams["min_samples_split"] = int(min_samples_split)
#         elif selected_model in ["Random Forest Regressor", "Random Forest Classifier"]:
#             n_estimators = st.number_input("Number of trees", min_value=10, value=100)
#             max_depth = st.number_input("Max depth", min_value=1, value=5)
#             min_samples_split = st.number_input("Min samples split", min_value=2, value=2)
#             hyperparams["n_estimators"] = int(n_estimators)
#             hyperparams["max_depth"] = int(max_depth)
#             hyperparams["min_samples_split"] = int(min_samples_split)
#         elif selected_model == "XGBoost":
#             n_estimators = st.number_input("Number of trees", min_value=10, value=100)
#             learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
#             max_depth = st.number_input("Max depth", min_value=1, value=3)
#             hyperparams["n_estimators"] = int(n_estimators)
#             hyperparams["learning_rate"] = float(learning_rate)
#             hyperparams["max_depth"] = int(max_depth)

#         if st.button("Train Model"):
#             try:
#                 if selected_model == "Linear Regression":
#                     pipe, X_test, y_test = linear_regression_train(df, target_col)
#                 elif selected_model == "Lasso Regression":
#                     pipe, X_test, y_test = lasso_train(df, target_col, **hyperparams)
#                 elif selected_model == "Ridge Regression":
#                     pipe, X_test, y_test = ridge_train(df, target_col, **hyperparams)
#                 elif selected_model == "Logistic Regression":
#                     pipe, X_test, y_test = logistic_regression_train(df, target_col, **hyperparams)
#                 elif selected_model == "Decision Tree Classifier":
#                     pipe, X_test, y_test = decision_tree_classifier_train(df, target_col, **hyperparams)
#                 elif selected_model == "Decision Tree Regressor":
#                     pipe, X_test, y_test = decision_tree_regressor_train(df, target_col, **hyperparams)
#                 elif selected_model == "Random Forest Classifier":
#                     pipe, X_test, y_test = random_forest_classifier_train(df, target_col, **hyperparams)
#                 elif selected_model == "Random Forest Regressor":
#                     pipe, X_test, y_test = random_forest_regressor_train(df, target_col, **hyperparams)
#                 elif selected_model == "XGBoost":
#                     pipe, X_test, y_test = xgboost_train(df, target_col, **hyperparams)
#                 else:
#                     st.warning("Selected model not implemented yet")
#                     pipe, X_test, y_test = None, None, None

#                 if pipe is not None:
#                     st.markdown(f'<div class="train-model">Trained {selected_model} successfully!</div>', unsafe_allow_html=True)

#                     st.session_state['trained_model'] = {"pipe": pipe, "X_test": X_test, "y_test": y_test}
#             except Exception as e:
#                 st.error(f"Error training {selected_model}: {e}")
        
#         else:
#             st.warning("Please upload the file first!")


# ------------------- Testing Tab -------------------
# elif st.session_state.active_tab == "🔍 Testing":
#     if 'trained_model' in st.session_state:
#         model_data = st.session_state['trained_model']
#         pipe = model_data['pipe']
#         X_test = model_data['X_test']
#         y_test = model_data['y_test']

#         problem_type = "regression" if np.issubdtype(y_test.dtype, np.number) else "classification"

#         if problem_type == "regression":
#             from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#             y_pred = pipe.predict(X_test)
#             st.markdown('<div class="eval-heading tab-separator">Regression Metrics</div>', unsafe_allow_html=True)
#             mae = mean_absolute_error(y_test, y_pred)
#             mse = mean_squared_error(y_test, y_pred)
#             score = r2_score(y_test, y_pred)
#             st.markdown(f"""
#                 <div class="metrics-grid">
#                     <div class="metrics-card">
#                         <div class="title">MAE</div>
#                         <div class="value">{mae:.2f}</div>
#                     </div>
#                     <div class="metrics-card">
#                         <div class="title">MSE</div>
#                         <div class="value">{mse:.2f}</div>
#                     </div>
#                     <div class="metrics-card">
#                         <div class="title">R² Score</div>
#                         <div class="value">{score:.2f}</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#         else:
#             from sklearn.metrics import accuracy_score, classification_report
#             # st.subheader("Classification Metrics")
#             st.markdown('<div class="eval-heading tab-separator">Classification Metrics</div>', unsafe_allow_html=True)
#             y_pred = pipe.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             st.markdown(f"""
#                 <div class="metrics-grid">
#                     <div class="metrics-card">
#                         <div class="title">Accuracy</div>
#                         <div class="value">{accuracy:.2f}</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             # st.metric("Accuracy", f"{accuracy:.2f}")
#             st.markdown('<div class="eval-report tab-separator">Classification Report</div>', unsafe_allow_html=True)
#             report = classification_report(y_test, y_pred, output_dict=True)
#             report_df = pd.DataFrame(report).transpose()
#             st.dataframe(report_df)
#     else:
#         st.warning("No trained model found. Please train a model in the 'Model Training' tab")
# ------------------- Testing Tab -------------------
elif st.session_state.active_tab == "🔍 Testing":
    model_data = st.session_state.trained_model

    if model_data is not None:
        pipe = model_data['pipe']
        X_test = model_data['X_test']
        y_test = model_data['y_test']

        problem_type = "regression" if np.issubdtype(y_test.dtype, np.number) else "classification"

        if problem_type == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = pipe.predict(X_test)
            st.markdown('<div class="eval-heading tab-separator">Regression Metrics</div>', unsafe_allow_html=True)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            score = r2_score(y_test, y_pred)
            st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metrics-card">
                        <div class="title">MAE</div>
                        <div class="value">{mae:.2f}</div>
                    </div>
                    <div class="metrics-card">
                        <div class="title">MSE</div>
                        <div class="value">{mse:.2f}</div>
                    </div>
                    <div class="metrics-card">
                        <div class="title">R² Score</div>
                        <div class="value">{score:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            from sklearn.metrics import accuracy_score, classification_report
            st.markdown('<div class="eval-heading tab-separator">Classification Metrics</div>', unsafe_allow_html=True)
            y_pred = pipe.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metrics-card">
                        <div class="title">Accuracy</div>
                        <div class="value">{accuracy:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div class="eval-report tab-separator">Classification Report</div>', unsafe_allow_html=True)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

    else:
        st.warning("No trained model found. Please train a model in the 'Model Training' tab")
