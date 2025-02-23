import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, PrecisionRecallDisplay)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# =======================================================================
# 1. Data Loading & JSON Parsing (Correct Implementation)
# =======================================================================
def load_and_preprocess_data(path, sample_size=100000):
    print("🔍 Loading and preprocessing data...")
    try:
        # Load data with proper encoding
        df = pd.read_csv(path, encoding='utf-8', nrows=sample_size)

        # Convert JSON-like columns from strings to dictionaries
        json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
        for col in json_cols:
            df[col] = df[col].apply(ast.literal_eval)

        # Flatten JSON columns
        device_df = pd.json_normalize(df['device'])
        geo_df = pd.json_normalize(df['geoNetwork'])
        traffic_df = pd.json_normalize(df['trafficSource'])
        totals_df = pd.json_normalize(df['totals'])

        # Merge all data
        df = pd.concat([
            df.drop(json_cols, axis=1),
            device_df.add_prefix('device.'),
            geo_df.add_prefix('geo.'),
            traffic_df.add_prefix('traffic.'),
            totals_df.add_prefix('totals.')
        ], axis=1)

        return df
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit()

# =======================================================================
# 2. Feature Engineering & Target Creation
# =======================================================================
def prepare_features_target(df):
    print("\n🎯 Preparing features and target...")

    # Create target: 1 if transactionRevenue > 0
    df['target'] = (pd.to_numeric(df['totals.transactionRevenue'], errors='coerce') > 0).astype(int)

    # Select meaningful features
    features = [
        'totals.hits',
        'totals.pageviews',
        'totals.timeOnSite',
        'device.isMobile',
        'geo.country',
        'traffic.source',
        'traffic.medium',
        'device.browser',
        'totals.sessionQualityDim'
    ]

    # Filter and clean
    df = df[features + ['target']].copy()
    df = df.replace({'not available in demo dataset': np.nan})

    return df.drop('target', axis=1), df['target']

# =======================================================================
# 3. Preprocessing Pipeline
# =======================================================================
def build_pipeline():
    # Numeric features
    numeric_features = ['totals.hits', 'totals.pageviews', 'totals.timeOnSite']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features
    categorical_features = ['device.isMobile', 'geo.country', 'traffic.source',
                            'traffic.medium', 'device.browser']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            class_weight='balanced_subsample',
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

# =======================================================================
# 4. Evaluation Metrics
# =======================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Evaluation Metrics:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Purchaser', 'Purchaser']))

    # Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt=',d',
                cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.show()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title('Precision-Recall Curve')
    plt.show()

# =======================================================================
# 5. Main Execution
# =======================================================================
if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'train_v2.csv'  # Update with your path
    SAMPLE_SIZE = 100000  # Adjust based on your system memory

    # Load and preprocess
    df = load_and_preprocess_data(DATA_PATH, SAMPLE_SIZE)
    X, y = prepare_features_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Build and train model
    pipeline = build_pipeline()
    print("\n🚀 Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    evaluate_model(pipeline, X_test, y_test)

    # Feature Importance
    feature_names = (
        numeric_features +
        list(pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_features))
    )

    importances = pipeline.named_steps['classifier'].feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})\
           .sort_values('importance', ascending=False)\
           .head(20)

    plt.figure(figsize=(12,8))
    sns.barplot(x='importance', y='feature', data=fi_df)
    plt.title('Top 20 Important Features')
    plt.show()
