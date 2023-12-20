import pandas as pd
import numpy as np


def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')
    return x.values, y.values


def prepare_data_loader(X, y, batch_size):
    n = X.shape[0]
    permutation = np.random.permutation(n)
    for start_idx in range(0, n, batch_size):
        end_idx = start_idx + batch_size if batch_size <= n else n
        batch_x = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        yield batch_x, batch_y

