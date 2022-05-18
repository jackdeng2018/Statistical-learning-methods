# -*- encoding: utf-8 -*-
'''
@File    :   logistic_regression.py
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 13:40   Deng      1.0         None
'''

# data
from math import exp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]


class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x_train):
        """
        sigmoid函数

        Args:
            x_train ():

        Returns:

        """
        return 1 / (1 + exp(-x_train))

    def data_matrix(self, x_train):
        """
        for循环 可以直接用矩阵运算

        Args:
            x_train ():

        Returns:

        """
        data_mat = []
        for d in x_train:
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, x_train, y_train):
        # label = np.mat(y)
        data_mat = self.data_matrix(x_train)  # m*n
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(x_train)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y_train[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, x_test, y_test):
        right = 0
        x_test = self.data_matrix(x_test)
        for x, y in zip(x_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(x_test)


def logistic_regression():
    x_in, y_in = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.3)
    lr_clf = LogisticReressionClassifier()
    lr_clf.fit(x_train, y_train)
    lr_clf.score(x_test, y_test)

    x_ponits = np.arange(4, 8)
    y_ = -(lr_clf.weights[1] * x_ponits + lr_clf.weights[0]) / lr_clf.weights[2]
    plt.plot(x_ponits, y_)

    # lr_clf.show_graph()
    plt.scatter(x_in[:50, 0], x_in[:50, 1], label='0')
    plt.scatter(x_in[50:, 0], x_in[50:, 1], label='1')
    plt.legend()
    plt.show()


def sk_lr():
    x_in, y_in = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.3)
    clf = LogisticRegression(max_iter=200)
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    print(clf.coef_, clf.intercept_)
    x_ponits = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.plot(x_in[:50, 0], x_in[:50, 1], 'bo', color='blue', label='0')
    plt.plot(x_in[50:, 0], x_in[50:, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    logistic_regression()
    sk_lr()
