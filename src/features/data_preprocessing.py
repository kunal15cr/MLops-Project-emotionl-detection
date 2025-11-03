# ...existing code...
import numpy as np
import pandas as pd

import os
import sys
import logging

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer , WordNetLemmatizer
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

# fetch data from data/raw
def load_data(file_path):
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        # ensure strings and no NaNs
        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace=True)
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error("Raw data files not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Raw data files are empty or invalid: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading data: %s", e)
        raise

# ...existing code...

def lammatization(text):
    try:
        lammatizer= WordNetLemmatizer()
        text = str(text).split()
        text = [lammatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception:
        return ""

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception:
        return ""

def removing_number(text):
    try:
        text = ''.join([i for i in str(text) if not i.isdigit()])
        return text
    except Exception:
        return ""

def lower_case(text):
    try:
        text = str(text).split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception:
        return ""

def removing_punctuations(text):
    try:
        text = str(text).replace(":", "")
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""

def remove_url(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', str(text))
    except Exception:
        return ""

def remove_small_sentence(df, column_name):
    try:
        df = df[df[column_name].apply(lambda x: len(str(x).split()) >= 3)]
        return df
    except Exception:
        logger.exception("Failed to remove small sentences from column %s", column_name)
        return df

def normalization_text(df):
    try:
        # defensive: ensure column exists and is string
        if "content" not in df.columns:
            logger.warning("Missing 'content' column, creating empty column")
            df["content"] = ""

        df["content"] = df["content"].fillna("").astype(str)

        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(removing_number)
        df["content"] = df["content"].apply(removing_punctuations)
        df["content"] = df["content"].apply(remove_url)
        df["content"] = df["content"].apply(lammatization)
        return df
    except Exception:
        logger.exception("Error during text normalization")
        # return at least original dataframe (with content column coerced to string)
        df["content"] = df.get("content", "").astype(str)
        return df

def proseass_data(train_data, test_data):
    try:
        train_processed_data = normalization_text(train_data)
        test_processed_data = normalization_text(test_data)
        return train_processed_data, test_processed_data
    except Exception:
        logger.exception("Error processing train/test data")
        raise

def save_data(train_processed_data, test_processed_data):
    data_path = os.path.join("data","processed")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)
    except OSError:
        logger.exception("Failed to create directory or write files to %s", data_path)
        raise
    except Exception:
        logger.exception("Unexpected error while saving processed data")
        raise

def main():
    try:
        #transform data
        try:
            nltk.download("wordnet")
            nltk.download("stopwords")
        except Exception:
            logger.warning("Failed to download NLTK data; assuming available locally")

        train_data, test_data = load_data("./data/raw/tweet_emotions.csv")

        train_processed_data, test_processed_data = proseass_data(train_data, test_data)

        save_data(train_processed_data, test_processed_data)

    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...