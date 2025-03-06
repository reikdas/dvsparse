import os
import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from bench import naive_path, dense_path, mkl_path

BASE_PATH=pathlib.Path(__file__).resolve().parent

if __name__ == "__main__":
    naive_df = pd.read_csv(naive_path)
    dense_df = pd.read_csv(dense_path)
    mkl_df = pd.read_csv(mkl_path)

    speedup_naive = naive_df['time'] / dense_df['time']
    speedup_mkl = mkl_df['time'] / dense_df['time']

    dense_df['speedup_naive'] = speedup_naive
    dense_df['speedup_mkl'] = speedup_mkl
    dense_df['size'] = dense_df['side'] * dense_df['side']

    dense_df['target'] = ((dense_df['speedup_naive'] > 1) & (dense_df['speedup_mkl'] > 1)).astype(int)

    # Define features (X) and target (y)
    X = dense_df[['size', 'density']]
    y = dense_df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train.values, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    model_filename = os.path.join(BASE_PATH, "models", "density_threshold_spmv.pkl")
    joblib.dump(model, model_filename)
