# ...existing code...
import sys
import logging
import yaml
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

control_handeler = logging.StreamHandler(sys.stdout)
control_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
control_handeler.setFormatter(formatter)
logger.addHandler(control_handeler)

def load_params(param_path: str) -> int:
    try:
        with open(param_path, "r") as fh:
            params = yaml.safe_load(fh) or {}
        max_features = int(params.get("feature_engineering", {}).get("max_features", 5000))
        return max_features
    except FileNotFoundError:
        logger.warning("params.yaml not found at %s, using default max_features=5000", param_path)
        return 5000
    except (yaml.YAMLError, ValueError) as e:
        logger.exception("Failed to read/parse params.yaml: %s", e)
        raise

def load_data():
    try:
        train_path = "./data/processed/train_processed.csv"
        test_path = "./data/processed/test_processed.csv"
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Processed files missing under data/processed")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        # ensure columns exist and coerce content to string
        train_data["content"] = train_data.get("content", "").fillna("").astype(str)
        test_data["content"] = test_data.get("content", "").fillna("").astype(str)
        return train_data, test_data
    except Exception:
        logger.exception("Failed to load processed data")
        raise

def data_vectorization(train_data, test_data, max_features):
    try:
        # defensive coercion
        train_data["content"] = train_data["content"].fillna("").astype(str)
        test_data["content"] = test_data["content"].fillna("").astype(str)

        x_train = train_data["content"].values
        y_train = train_data["sentiment"].values if "sentiment" in train_data.columns else None

        x_test = test_data["content"].values
        y_test = test_data["sentiment"].values if "sentiment" in test_data.columns else None

        if len([s for s in x_train if s and str(s).strip()]) == 0:
            raise RuntimeError("No non-empty training documents found in processed training data")

        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)
        feature_names = vectorizer.get_feature_names_out()
        return x_train_bow, y_train, x_test_bow, y_test, feature_names
    except Exception:
        logger.exception("Vectorization failed")
        raise

def mearg_data(x_train_bow, y_train, x_test_bow, y_test, feature_names):
    try:
        train_df = pd.DataFrame(x_train_bow.toarray(), columns=feature_names)
        if y_train is not None:
            train_df["sentiment"] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray(), columns=feature_names)
        if y_test is not None:
            test_df["sentiment"] = y_test
        return train_df, test_df
    except Exception:
        logger.exception("Failed to merge matrices into DataFrame")
        raise

def save_data(train_df, test_df):
    data_path = os.path.join("data", "features")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logger.info("Feature files written to %s", data_path)
    except Exception:
        logger.exception("Failed to save feature files to %s", data_path)
        raise

def main():
    try:
        max_features = load_params("params.yaml")
        train_data, test_data = load_data()
        x_train_bow, y_train, x_test_bow, y_test, feature_names = data_vectorization(train_data, test_data, max_features)
        train_df, test_df = mearg_data(x_train_bow, y_train, x_test_bow, y_test, feature_names)
        save_data(train_df, test_df)
    except Exception:
        logger.exception("Feature engineering pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...