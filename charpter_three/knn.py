# -*- encoding: utf-8 -*-
'''
@File    :   knn.py   
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/13 14:30   Deng      1.0         None
'''

import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter


def length(x_train, y_train, p_train=2):
    """
    计算距离
    p = 1 曼哈顿距离
    p = 2 欧氏距离
    p = inf 闵式距离 minkowski_distance

    Args:
        x_train ():
        y_train ():
        p_train ():

    Returns:

    """
    if len(x_train) == len(y_train) and len(x_train) > 1:
        sum_length = 0
        for index in range(len(x_train)):
            sum_length += math.pow(abs(x_train[index] - y_train[index]), p_train)
        return math.pow(sum_length, 1 / p_train)
    else:
        return 0


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
    return df


class KNN:
    """
    k 最近邻算法
    """

    def __init__(self, x_train, y_train, n_neighbors=3, p_train=2):
        """
        Args:
            x_train (): x
            y_train (): y
            n_neighbors (): 临近点个数
            p_train (): 距离度量
        """
        self.neighbors = n_neighbors
        self.p_train = p_train
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_point):
        """
        预测
        Args:
            x_point (): 新点

        Returns:

        """
        # 取出n个点
        knn_list = []
        for i in range(self.neighbors):
            # 求范数
            dist = np.linalg.norm(x_point - self.x_train[i], ord=self.p_train)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.neighbors, len(self.x_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(x_point - self.x_train[i], ord=self.p_train)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])


if __name__ == '__main__':
    # 距离计算
    # 课本例3.1
    x1 = [1, 1]
    x2 = [5, 1]
    x3 = [4, 4]
    # x1, x2
    for index in range(1, 5):
        r = {'1-{}'.format(c): length(x1, c, p_train=index) for c in [x2, x3]}
        # print(min(zip(r.values(), r.keys())))

    # knn
    data_frame = load_data()
    data = np.array(data_frame.iloc[:100, [0, 1, -1]])
    x_data, y_data = data[:, :-1], data[:, -1]
    # 没用交叉验证集 只用了测试集
    x_tr, x_test, y_tr, y_test = train_test_split(x_data, y_data, test_size=0.2)
