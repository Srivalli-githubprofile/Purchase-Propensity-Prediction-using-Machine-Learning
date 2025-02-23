Purchase Propensity Prediction using Machine Learning
Description:
This project builds a Random Forest Classifier to predict a user's likelihood of making a purchase based on website interaction data. The model is trained on Google Analytics session data and utilizes feature engineering, SMOTE for handling class imbalance, and performance evaluation metrics.

Project Structure
load_and_preprocess_data(path, sample_size):

Loads the dataset and parses JSON-like columns.
Flattens nested JSON structures for better feature extraction.
Handles missing or unavailable data.
prepare_features_target(df):

Creates a binary target variable (target) indicating whether a user made a purchase (transactionRevenue > 0).
Selects key behavioral and categorical features like session duration, page views, and traffic source.
build_pipeline():

Constructs a preprocessing pipeline with One-Hot Encoding, Imputation, and Standard Scaling.
Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
Trains a Random Forest Classifier with optimized hyperparameters.
evaluate_model(model, X_test, y_test):

Evaluates model performance using ROC AUC, Precision-Recall Curve, and Confusion Matrix.
Generates visualizations for classification performance and feature importance.


Technologies & Libraries Used
Python 3.x
Pandas, NumPy (Data Manipulation)
Scikit-learn (Model Training & Preprocessing)
Imbalanced-learn (SMOTE) (Handling Class Imbalance)
Matplotlib & Seaborn (Visualization)


Results & Insights
The Random Forest Model effectively classifies potential buyers vs. non-buyers.
Top Features Influencing Purchase Propensity:
Session quality score
Page views per session
Time spent on the website
Traffic source & Medium
