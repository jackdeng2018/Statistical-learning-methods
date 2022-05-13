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
from sklearn.model_selection import train_test_split


def load_data():
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target

    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    df.label.value_counts()

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
    return np.array(df.iloc[:100, [0, 1, -1]])


def pandas_perceptron(data, X, y):
    """
    自己写的感知机

    Args:
        X ():
        y ():

    Returns:

    """
    perceptron_origin = PerceptronOrigin(data)
    perceptron_origin.fit(X, y)
    # 超平面可视化
    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron_origin.w_train[0] * x_points + perceptron_origin.b_train) / perceptron_origin.w_train[1]
    print(perceptron_origin.w_train)
    plt.plot(x_points, y_)
    plot_data(data)
    return perceptron_origin.w_train, perceptron_origin.b_train


def plot_data(data):
    """
    数据展示

    Args:
        data ():

    Returns:

    """
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def dual_perceptron(data):
    """
    感知机对偶形式

    Args:
        data (): 数据源

    Returns:

    """
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    perceptron_dual = PerceptronDual()
    perceptron_dual.fit(X, y)
    # 超平面可视化
    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron_dual.weight[0] * x_points + perceptron_dual.b_train) / perceptron_dual.weight[1]
    print(perceptron_dual.weight)
    plt.plot(x_points, y_)

    plot_data(data)
    return perceptron_dual.weight, perceptron_dual.b_train


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
        self.b_train = 0.0
        # 定义学习率 0到1
        self.l_rate = 0.1

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
        train_times = 0
        while not is_wrong:
            train_times += 1
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
                    for index in range(len(x_train)):
                        x_in = x_train[index]
                        y_in = y_train[index]
                        # print(y_in * self.sign(x_in, self.w_train, self.b_train))
                        if y_in * self.sign(x_in, self.w_train, self.b_train) <= 0:
                            is_wrong = False
                            break

        print('Perceptron Model train times :' + str(train_times))

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
        self.l_rate = 0.1

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
        train_times = 0
        while not train_complete_flag:
            train_times += 1
            error_count = 0
            for i in range(x_train.shape[0]):
                y_in = y_train[i]
                # 有—个点当分类错误时,更新alpha_i和偏置 都是矩阵运算
                tmp_sum = np.multiply(self.alpha, y_train) @ gram[:, i]
                # 进行误分类点判断
                if y_in * (tmp_sum + self.b_train) <= 0:
                    self.alpha[i] += self.l_rate
                    self.b_train += self.l_rate * y_in
                    error_count += 1
            if not error_count:
                train_complete_flag = True  # 训练完成后计算权重
                self.weight = np.multiply(self.alpha, y_train) @ x_train
                print('train times:' + str(train_times))


def score(test_x, test_y, weights, b_0):
    """
    计算得分

    Args:
        test_x ():
        test_y ():

    Returns:

    """
    correct_count = 0
    for index in range(len(test_x)):
        x_i = test_x[index]
        y_i = test_y[index]
        if y_i * ((weights @ x_i) + b_0) >= 0:
            correct_count += 1
    return correct_count / len(test_x)

if __name__ == '__main__':
    # 加载数据
    iris_data = load_data()
    X, y = iris_data[:, :-1], iris_data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    # 划分训练集，交叉验证集，测试集
    train_feature, test_feature, train_target, test_target = train_test_split(
        X, y, test_size=0.2, random_state=56)

    cross_feature, test_x, cross_target, test_y = train_test_split(test_feature, test_target, test_size=0.5,
                                                                   random_state=56)

    # 原始形式
    w_1, b_1 = pandas_perceptron(iris_data, train_feature, train_target)

    print(str(score(cross_feature, cross_target, w_1, b_1)))
    print(str(score(test_x, test_y, w_1, b_1)))

    # 对偶形式
    w_1, b_1 = dual_perceptron(iris_data)
    print(str(score(cross_feature, cross_target, w_1, b_1)))
    print(str(score(test_x, test_y, w_1, b_1)))

    # scikit 感知机
    # scikit_learn_perceptron(iris_data)
    # print(str(score(cross_feature, cross_target, w_1, b_1)))
    # print(str(score(test_x, test_y, w_1, b_1)))
