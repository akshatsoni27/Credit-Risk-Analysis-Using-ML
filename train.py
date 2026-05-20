import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
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
    
    # Feature engineering: Add interaction features and domain-specific features
    X['Age_CreditRatio'] = X['Age'] / (X['Credit amount'] / 1000 + 1)
    X['Duration_CreditRatio'] = X['Duration'] / (X['Credit amount'] / 1000 + 1)
    X['CreditAmount_Duration'] = X['Credit amount'] / (X['Duration'] + 1)
    X['Age_Duration'] = X['Age'] / (X['Duration'] + 1)
    X['Credit_per_month'] = X['Credit amount'] / (X['Duration'] + 1)

    # Define features
    numeric_features = ['Age', 'Credit amount', 'Duration', 'Age_CreditRatio', 'Duration_CreditRatio', 'CreditAmount_Duration', 'Age_Duration', 'Credit_per_month']
    categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    # Convert Job to string so it's treated as categorical
    X['Job'] = X['Job'].astype(str)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    print("Training optimized LightGBM model for accuracy...")

    # Highly optimized LightGBM for maximum accuracy
    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        num_leaves=60,
        max_depth=12,
        learning_rate=0.015,
        n_estimators=400,
        subsample=0.95,
        colsample_bytree=0.95,
        class_weight='balanced',
        is_unbalanced=True,
        metric='binary_logloss',
        verbose=-1,
        min_split_gain=0.0,
        min_child_weight=1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        path_smooth=0.0
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb_model)
    ])

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Test Set Accuracy: {test_accuracy:.2%}")
    print(f"{'='*50}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the pipeline
    model_filename = 'model.joblib'
    joblib.dump(pipeline, model_filename)
    print(f"\nModel successfully saved to {model_filename}")

if __name__ == "__main__":
    main()
