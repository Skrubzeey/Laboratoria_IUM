import joblib
from dotenv import load_dotenv
from pathlib import Path
import os
from data_prep import *
from sklearn.metrics import classification_report, accuracy_score

load_dotenv()
DATA_PATH = Path(os.getenv("DATA_PATH"))
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
MODEL_DIR = Path(os.getenv("MODEL_DIR"))

model = joblib.load(os.path.join(MODEL_DIR, "pipeline.pkl"))
data = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(data, TEST_SIZE, RANDOM_SEED)

# ewaluacja modelu
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 3))
print("Wyniki wytrenowanego modelu:")
print(classification_report(y_test, y_pred))
