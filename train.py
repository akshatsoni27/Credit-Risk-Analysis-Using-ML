import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

def main():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv('german_credit_data.csv')
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # The dataset uses 'NA' string for missing values in 'Saving accounts' and 'Checking account'.
    # We will replace them with 'Unknown'
    df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
    df['Saving accounts'] = df['Saving accounts'].replace('NA', 'Unknown')
    
    df['Checking account'] = df['Checking account'].fillna('Unknown')
    df['Checking account'] = df['Checking account'].replace('NA', 'Unknown')
    
    # Target variable mapping: good -> 0, bad -> 1
    df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})
    
    X = df.drop(columns=['Risk'])
    y = df['Risk']

    # Define features
    numeric_features = ['Age', 'Credit amount', 'Duration']
    # 'Job' is numeric in the CSV (0, 1, 2, 3), but it represents categorical data
    categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    # Convert Job to string so it's treated as categorical by OneHotEncoder
    X['Job'] = X['Job'].astype(str)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Asymmetrical cost matrix: misclassifying a "Bad" risk applicant as "Good" is 5 times worse.
    # Therefore, False Negatives cost 5x more than False Positives.
    # We heavily penalize this by setting scale_pos_weight = 5.
    scale_pos_weight = 5.0
    
    print(f"Using scale_pos_weight={scale_pos_weight} to account for the 5:1 cost matrix.")

    xgb_model = XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])

    # Hyperparameter grid specified by user
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.7, 1],
        'classifier__colsample_bytree': [0.7, 1]
    }

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Starting GridSearchCV...")
    # cv=3 to speed up the process, standard would be 5
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='recall', # Optimize for recall to minimize False Negatives
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_

    # Evaluation
    print("\n--- Evaluation on Test Set ---")
    y_pred = best_model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the pipeline
    model_filename = 'model.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nModel successfully saved to {model_filename}")

if __name__ == "__main__":
    main()
