import pandas as pd
import numpy as np
import os

# funkcja do zapisania danych w DataFrame
def load_data(path):
    data = pd.read_csv(os.path.join(path, "diabetes.csv"))
    return data


# funkcja do podziału danych
def split_data(df, test_size, seed):
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    n = len(y)
    split = int(n*test_size)
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(y))

    train_indices = indices[:split]
    test_indices = indices[split:]

    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return x_train, x_test, y_train, y_test