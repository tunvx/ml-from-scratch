from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from data.load_data import sklearn_to_df
from supervised_learning import MyLinearRegression, MyLogisticRegression


# TEST FOR LOGISTIC REGRESSION
def test01_logistic_model():
    X, y = sklearn_to_df(load_breast_cancer())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 125
    model = LogisticRegression(solver='newton-cg', max_iter=n_iter)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy sklearn model: {accuracy}')

    model1 = MyLogisticRegression(lr=0.1, n_iter=n_iter)
    model1.fit(X_tr, y_tr)
    y_pred = model1.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'With {n_iter} iters, accuracy my model: {accuracy}')

# TEST FOR LINEAR REGRESSION
def test02_logistic_model():
    X, y = sklearn_to_df(load_diabetes())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_iter = 125
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


if __name__ == "__main__":
    test01_logistic_model()
    test02_logistic_model()
