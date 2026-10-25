import pandas as pd
import numpy as np

# funkcja do zapisania danych w DataFrame
def load_data(path):
    data = pd.read_csv(path)
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

    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test