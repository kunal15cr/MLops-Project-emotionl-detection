# ...existing code...
import numpy as np
import pandas as pd
import yaml
import logging
import os
import sys

from sklearn.ensemble import GradientBoostingClassifier

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

control_handeler = logging.StreamHandler(sys.stdout)
control_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
control_handeler.setFormatter(formatter)
logger.addHandler(control_handeler)

def load_params(param_path: str) -> dict:
    try:
        with open(param_path, "r") as fh:
            params = yaml.safe_load(fh)
        mb = params.get('model_building')
        if not isinstance(mb, dict):
            raise KeyError("Missing 'model_building' section in params.yaml")
        # validate required keys
        if 'n_estimators' not in mb or 'learning_rate' not in mb:
            raise KeyError("params 'n_estimators' and 'learning_rate' required under 'model_building'")
        return mb
    except FileNotFoundError:
        logger.exception("params.yaml not found at %s", param_path)
        raise
    except yaml.YAMLError:
        logger.exception("Failed to parse YAML file: %s", param_path)
        raise
    except Exception:
        logger.exception("Unexpected error loading params")
        raise

def load_data():
    try:
        train_data = pd.read_csv("./data/features/train_bow.csv")
        test_data = pd.read_csv("./data/features/test_bow.csv")
    except FileNotFoundError:
        logger.exception("Feature files not found under data/features")
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Feature files are empty or corrupted")
        raise
    except Exception:
        logger.exception("Unexpected error reading feature files")
        raise

    if "sentiment" not in train_data.columns:
        logger.error("'sentiment' column missing from train_bow.csv")
        raise KeyError("'sentiment' column missing")

    try:
        x_train = train_data.drop("sentiment", axis=1).values
        y_train = train_data["sentiment"].values
    except Exception:
        logger.exception("Failed to prepare training arrays")
        raise

    if x_train.size == 0 or y_train.size == 0:
        logger.error("Empty training data")
        raise ValueError("Empty training data")

    return x_train, y_train

def train_model(x_train, y_train, params):
    try:
        n_estimators = int(params['n_estimators'])
        learning_rate = float(params['learning_rate'])
    except Exception:
        logger.exception("Invalid hyperparameters in params")
        raise

    try:
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42
        )
        clf.fit(x_train, y_train)
        return clf
    except Exception:
        logger.exception("Model training failed")
        raise

def save_model(clf):
    import joblib
    model_path = "./data/gradient_boosting_model.pkl"
    try:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(clf, model_path)
        logger.info("Model saved to %s", model_path)
    except Exception:
        logger.exception("Failed to save model to %s", model_path)
        raise

def main():
    try:
        params = load_params("params.yaml")
        x_train, y_train = load_data()
        clf = train_model(x_train, y_train, params)
        save_model(clf)
    except Exception:
        logger.exception("Model building pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...