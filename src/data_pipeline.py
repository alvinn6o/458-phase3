"""
Data pipeline file to read dataset and vectorize
tf-idf vectorize from sklearn to convert the text into features
it tokenizes documents automatically for our ML models
splits 80/10/10 set
SAVE as npz (compressed format of features from tfidf) so we dont have to run every time

this is first step the process follows like
data pipeline (once) --> training models (saves the models after) --> evaluate to see metrics --> predict which used in app.py to load everything

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 10


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load
    df = pd.read_csv(os.path.join(DATA_DIR, "phishing_email.csv"))
    df = df.rename(columns={"text_combined": "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Dataset: {len(df)} rows")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    # 80/10/10 split
    train, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE, stratify=temp["label"])
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # TF-IDF 
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.95,
        min_df=5,
    )
    X_train = tfidf.fit_transform(train["text"].fillna(""))
    X_val = tfidf.transform(val["text"].fillna(""))
    X_test = tfidf.transform(test["text"].fillna(""))
    print(f"Features: {X_train.shape[1]}")

    # Save
    save_npz(os.path.join(DATA_DIR, "X_train.npz"), X_train)
    save_npz(os.path.join(DATA_DIR, "X_val.npz"), X_val)
    save_npz(os.path.join(DATA_DIR, "X_test.npz"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), train["label"].values)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), val["label"].values)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), test["label"].values)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    print("end of data pipelin")


if __name__ == "__main__":
    main()
