# -*- encoding: utf-8 -*-
'''
@File    :   perceptron.py   
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/12 16:16   Deng      1.0         None
'''

# 算法 随即梯度下降法 Stochastic Gradient Descent
# 随机抽取一个误分类点使其梯度下降。
# 当实例点被误分类，即位于分离超平面的错误侧，则调整w, b的值，使分离超平面向该无误类点的一侧移动，直至误分类点被正确分类

# 拿出iris数据集中两个分类的数据和[sepal length，sepal width]作为特征

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


def load_data():
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target

    #
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    df.label.value_counts()

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
    return np.array(df.iloc[:100, [0, 1, -1]])


def pandas_perceptron(data):
    """
    自己写的感知机

    Args:
        data (): 数据

    Returns:

    """
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    perceptron_origin = PerceptronOrigin(data)
    perceptron_origin.fit(X, y)

    # 超平面可视化
    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron_origin.w_train[0] * x_points + perceptron_origin.b_train) / perceptron_origin.w_train[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def scikit_learn_perceptron(data):
    """
    scikit_learn包里的感知机

    Args:
        data (): 数据集

    Returns:

    """
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(X, y)
    # Weights assigned to the features.
    print(clf.coef_)
    # 截距 Constants in decision function.
    print(clf.intercept_)
    # 计算分数
    print(clf.score(X, y))

    x_ponits = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


# 感知机--原始算法
class PerceptronOrigin:
    # 构造时值初始化 w = (0, 0) b = 0
    def __init__(self, data):
        # 初始化参数 w , b,和学习率
        self.w_train = np.zeros(len(data[0]) - 1, dtype=np.float32)
        self.b_train = 0
        # 定义学习率 0到1
        self.l_rate = 1

    def sign(self, x_train, w_train, b_train):
        """
        感知机模型函数

        Args:
            x_train ():
            w_train ():
            b_train ():

        Returns:

        """
        return (w_train @ x_train) + b_train

    def fit(self, x_train, y_train):
        """
        随机梯度下降法

        Args:
            x_train ():
            y_train ():

        Returns:

        """
        # 判断是否有误分类点
        is_wrong = False
        while not is_wrong:
            # 记录误分类点数目
            wrong_count = 0
            for index in range(len(x_train)):
                # 选取—个点进行判断
                x_in = x_train[index]
                y_in = y_train[index]
                # 判断是不是误分类点的标志
                if y_in * self.sign(x_in, self.w_train, self.b_train) <= 0:
                    # 更新参数 y_ @ x_ 为loss func对w的导数  y_为loss func对b的导数
                    self.w_train = self.w_train + self.l_rate * np.dot(y_in, x_in)
                    self.b_train = self.b_train + self.l_rate * y_in
                    # 误分类点加1
                    wrong_count += 1

                # 没有误分类点
                if wrong_count == 0:
                    is_wrong = True
        return 'Perceptron Model'

    def score(self):
        pass


# 感知机-对偶算法
class PerceptronDual:
    # 构造时值初始化 w = (0, 0) b = 0
    def __init__(self):
        # 初始化参数 alpha[i] = n[i] * l_rate
        self.alpha = 0
        self.weight = 0
        self.b_train = 0
        # 学习率
        self.lr = 0.1

    def fit(self, x_train, y_train):
        """
        随机梯度下降法

        Args:
            x_train ():
            y_train ():

        Returns:

        """
        # 权重和偏置初始化
        self.alpha = np.zeros(x_train.shape[0])
        self.b_train = 0

        # 训练结束标志
        train_complete_flag = False

        # 存放样本两两内积的Gram矩阵（特点)
        gram = x_train @ x_train.T

        while not train_complete_flag:
            error_count = 0
            for i in range(x_train.shape[0]):
                x_, y_ = x_train[i], y_train[i]
                # 有—个点当分类错误时,更新alpha_i和偏置 都是矩阵运算
                tmp_sum = np.multiply(self.alpha, y_train) @ gram[:, i]
                # 进行误分类点判断
                if y_ * (tmp_sum + self.b_train) <= 0:
                    self.alpha[i] += self.lr
                    self.b_train += self.lr * y_
                    error_count += 1
            if not error_count:
                train_complete_flag = True  # 训练完成后计算权重
                self.weight = np.multiply(self.alpha, y_train) @ x_train


if __name__ == '__main__':
    iris_data = load_data()
    pandas_perceptron(iris_data)
    # scikit_learn_perceptron(iris_data)
