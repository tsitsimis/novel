import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def kernel_matrix(kernel, x):
    n = x.shape[0]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = kernel(x[i, :], x[j, :])
    return matrix


def cross_val_score(clf, X, y, k=5):
    X_good = X[y == 1]
    X_bad = X[y == 0]

    y_good = y[y == 1]
    y_bad = y[y == 0]

    kf = KFold(n_splits=k)
    acc_scores = np.zeros(k)
    i = 0
    for train_index, test_index in kf.split(X_good):
        X_train, X_test = X_good[train_index], X_good[test_index]
        X_test = np.concatenate((X_test, X_bad), axis=0)
        y_test = np.concatenate((y_good[test_index], y_bad), axis=0)

        # normalize
        X_train = StandardScaler().fit(X).transform(X_train)
        X_test = StandardScaler().fit(X).transform(X_test)

        clf.fit(X_train)
        # print("Number of support vectors: %d" % clf.support_vectors.shape[0])

        y_pred = clf.predict(X_test)
        acc_scores[i] = accuracy_score(y_test, y_pred)
        # print("Accuracy: %0.2f" % acc_scores[i])
        i += 1

    return acc_scores
