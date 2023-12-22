import pandas as pd
import numpy as np
import h5py


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


def load_catvnoncat_data():
    train_dataset = h5py.File('data/catvnoncat/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/catvnoncat/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes