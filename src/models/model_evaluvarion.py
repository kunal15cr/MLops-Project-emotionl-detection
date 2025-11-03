import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_test_data(test_path):
    """Load and prepare test data"""
    try:
        test_data = pd.read_csv(test_path)
        if 'sentiment' not in test_data.columns:
            raise ValueError("Test data missing 'target' column")
        X_test = test_data.drop('sentiment', axis=1)
        y_test = test_data['sentiment']
        logger.info("✅ Test data loaded successfully")
        return X_test, y_test
    except Exception as e:
        logger.error("Failed to load test data: %s", e)
        raise

def load_model(model_path):
    """Load trained model"""
    try:
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted'))
        }

        logger.info("Model Performance Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k.capitalize()}: {v:.4f}")

        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        return metrics
    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise

def save_metrics(metrics, metrics_path):
    """Save evaluation metrics"""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"✅ Metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error("Failed to save metrics: %s", e)
        raise

def main():
    try:
        test_path = os.path.join("data", "features", "test_bow.csv")
        model_path = os.path.join("data", "gradient_boosting_model.pkl")
        metrics_path = os.path.join("data", "metrics", "metrics.json")

        X_test, y_test = load_test_data(test_path)
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)

        logger.info("✅ Model evaluation completed successfully")
    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
