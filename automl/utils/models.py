
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from .utils import prob_detec  # function to detect regression/classification

# Generic preprocessing for any dataset
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Numeric & categorical columns
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    transformers = []
    if num_feat:
        transformers.append(("num", StandardScaler(), num_feat))
    if cat_feat:
        transformers.append(("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feat))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"
    return X, y, preprocessor



#  Linear & Regression Models 
def linear_regression_train(df, target_col):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test

def lasso_train(df, target_col, alpha=1.0):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([("preprocess", preprocessor), ("model", Lasso(alpha=alpha))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test

def ridge_train(df, target_col, alpha=1.0):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=alpha))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test



# logistic regression 
def logistic_regression_train(df, target_col, C=1.0, penalty='l2'):
    X, y, preprocessor = preprocess_data(df, target_col)
    if prob_detec(y) != "classification":
        raise ValueError("Target column is not suitable for classification")
    
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    
    return pipe, X_test, y_test



# Decision Tree Classifier
def decision_tree_classifier_train(df, target_col, **kwargs):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            criterion=kwargs.get("criterion", "gini"),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


# Decision Tree Regressor
def decision_tree_regressor_train(df, target_col, **kwargs):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeRegressor(
            criterion=kwargs.get("criterion", "squared_error"),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test




# Random Forest Classifier
def random_forest_classifier_train(df, target_col, **kwargs):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test



# Random Forest Regressor
def random_forest_regressor_train(df, target_col, **kwargs):
    X, y, preprocessor = preprocess_data(df, target_col)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test

# XGBoost Classifier
from sklearn.preprocessing import LabelEncoder

def xgboost_train(df, target_col, **kwargs):
    X, y, preprocessor = preprocess_data(df, target_col)
    
    # Encode target if classification
    if prob_detec(y) == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y) 

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.1),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    
    return pipe, X_test, y_test

