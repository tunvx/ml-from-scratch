import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_digits
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# Generate synthetic data
from sklearn.datasets import make_classification

import nn
from data.load_data import sklearn_to_df
from data.make_data import load_planar_dataset, load_extra_datasets
from nn import CrossEntropyLoss
from supervised_learning import *
from optim import SGDOptimizer


# TEST FOR LOGISTIC REGRESSION
def test01_logistic_model():
    X, y = sklearn_to_df(load_breast_cancer())
    print(X.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 125
    model = LogisticRegression(solver='newton-cg', max_iter=n_iter)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy sklearn model: {accuracy}')
    print(classification_report(y_te, y_pred))

    model1 = MyLogisticRegression(lr=0.1, n_iter=n_iter)
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy my model: {accuracy}')
    print(classification_report(y_te, y_pred))


# TEST FOR LINEAR REGRESSION
def test02_linear_model():
    X, y = sklearn_to_df(load_diabetes())
    print(X.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 500
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    loss = mean_squared_error(y_te, y_pred)
    print(f'Loss sklearn model: {loss}')

    model1 = MyLinearRegression(lr=0.1, n_iter=n_iter)
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    loss = mean_squared_error(y_te, y_pred)
    print(f'With {n_iter} iters, loss my model: {loss}')


def test03_multinomial_model():
    X, y = sklearn_to_df(load_iris())
    print(X.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 500
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=n_iter)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy sklearn model: {accuracy}')
    print(classification_report(y_te, y_pred))

    model1 = MyMultinomialRegression(lr=0.1, n_iter=n_iter)
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    loss = mean_squared_error(y_te, y_pred)
    print(f'With {n_iter} iters, loss my model: {loss}')
    print(classification_report(y_te, y_pred))


def test04_knn_classifier_model():
    X, y = sklearn_to_df(load_iris())
    print(X.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 500
    model = KNeighborsClassifier()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy sklearn model: {accuracy}')
    print(classification_report(y_te, y_pred))

    model1 = MyKNNClassifier(k=5)
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    loss = mean_squared_error(y_te, y_pred)
    print(f'With {n_iter} iters, loss my model: {loss}')
    print(classification_report(y_te, y_pred))


# TEST FOR LOGISTIC REGRESSION
def test05_mlp_binary_classifier():
    X, Y = load_planar_dataset()
    Y = Y.flatten()
    print(X.shape, Y.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.21)

    n_epochs = 25
    batch_size = 8

    # Instantiate the model
    model = MLPBinaryClassification(n_input=2, n_hidden=5, n_output=1)
    # Train the model
    model.fit(X_tr, y_tr, batch_size=batch_size, learning_rate=0.15, epochs=n_epochs)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_epochs} epochs, accuracy my MLP model: {accuracy}')
    print(classification_report(y_te, y_pred))

    model1 = MyMLPClassifier()
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_epochs} epochs, accuracy sklearn model: {accuracy}')
    print(classification_report(y_te, y_pred))


def test05_mlp_digits_classifier():
    X, Y = sklearn_to_df(load_digits())
    print(f"Original shape, X ({X.shape}), Y({Y.shape})")
    X, Xtest, y, ytest  = train_test_split(X, Y, test_size=0.33, random_state=42)
    transform = MinMaxScaler()
    X = transform.fit_transform(X)
    Xtest = transform.transform(Xtest)

    n_in, n_out = X.shape[1], 10

    model = MyMLPClassifier(n_input=n_in, hiddens=[128, 64, 32, 10], n_classes=n_out)
    model.info()
    params = model.parameters()
    grads = model.grads()
    print(len(params), len(grads))

    for i in range(10):
        print(params[i].shape, grads[i].shape, params[i].shape == grads[i].shape)

    optimizer = SGDOptimizer(model, learning_rate=0.1, regularization=0.03)
    criterion = nn.CrossEntropyLoss()

    # Huấn luyện mô hình theo từng mini-batch
    batch_size = 8
    num_epochs = 2
    num_samples = X.shape[0]

    loss = None

    for epoch in range(num_epochs):
        # Shuffle dữ liệu để tạo ra các mini-batch ngẫu nhiên
        indices = np.random.permutation(num_samples)

        step = 0
        total_loss, total_correct = 0, 0
        total_sample = 0

        for i in range(0, num_samples, batch_size):
            # Lấy mini-batch
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            batch_y = Y[batch_indices]

            # Forward pass
            batch_yp = model.forward(X_batch)
            loss = criterion.forward(batch_yp, batch_y)

            # Backward pass and an optimization step
            optimizer.zero_grad()
            dout = criterion.backward()
            dx = model.backward(dout)
            optimizer.step()

            ## log training progress
            step += 1
            total_loss += loss
            total_correct += np.sum(np.argmax(batch_yp, axis=1) == batch_y)
            total_sample += len(batch_y)
            print(f"epoch {epoch} step {step} train_loss {total_loss / total_sample:.4f} train_acc {total_correct / total_sample:.4f}")

    ypred = np.argmax(model.forward(Xtest), axis=1)
    print(classification_report(ytest, ypred))
    print(confusion_matrix(ytest, ypred))

if __name__ == "__main__":
    # test01_logistic_model()
    # test02_linear_model()
    # test03_multinomial_model()
    # test04_knn_classifier_model()
    # test05_mlp_binary_classifier()
    test05_mlp_digits_classifier()

