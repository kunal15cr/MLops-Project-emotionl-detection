# ...existing code...
import os
import sys
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

control_handeler = logging.StreamHandler(sys.stdout)
control_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
control_handeler.setFormatter(formatter)
logger.addHandler(control_handeler)

def load_params(param_path: str = "params.yaml") -> float:
    try:
        with open(param_path, "r") as fh:
            params = yaml.safe_load(fh) or {}
        section = params.get("data_ingation") or params.get("data_ingestion") or {}
        test_size = float(section.get("test_size", 0.2))
        return test_size
    except FileNotFoundError:
        logger.warning("%s not found, using default test_size=0.2", param_path)
        return 0.2
    except Exception:
        logger.exception("Failed to load params from %s", param_path)
        raise

def read_data() -> pd.DataFrame:
    try:
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv"
        df = pd.read_csv(url)
        return df
    except Exception:
        logger.exception("Failed to read source data")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        if "tweet_id" in df.columns:
            df.drop(columns="tweet_id", inplace=True)
        df_final = df[df["sentiment"].isin(["sadness", "neutral"])].copy()
        df_final["sentiment"] = df_final["sentiment"].map({"neutral": 1, "sadness": 0}).astype(int)
        return df_final
    except Exception:
        logger.exception("Failed to preprocess data")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, out_dir: str = "./data/raw"):
    try:
        os.makedirs(out_dir, exist_ok=True)
        train_data.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(out_dir, "test.csv"), index=False)
        logger.info("Saved raw train/test to %s", out_dir)
    except Exception:
        logger.exception("Failed to save raw data")
        raise

def main():
    try:
        params = load_params("params.yaml")
        df = read_data()
        df_final = preprocess_data(df)
        train_data, test_data = train_test_split(df_final, test_size=params, random_state=42, stratify=df_final.get("sentiment"))
        save_data(train_data, test_data)
    except Exception:
        logger.exception("Data ingestion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...