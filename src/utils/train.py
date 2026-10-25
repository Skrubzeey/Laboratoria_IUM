import joblib
from dotenv import load_dotenv
from pathlib import Path
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from data_prep import *

def train_pipeline(path, size, seed, output_path):
    data = load_data(path=path)
    X_train, X_test, y_train, y_test = split_data(data, size, seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(output_path, "pipeline.pkl"))


# urchomienie skryptu
load_dotenv()
DATA_PATH = Path(os.getenv("DATA_PATH"))
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
MODEL_DIR = Path(os.getenv("MODEL_DIR"))

train_pipeline(DATA_PATH, TEST_SIZE, RANDOM_SEED, MODEL_DIR)