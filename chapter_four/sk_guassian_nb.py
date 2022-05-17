# -*- encoding: utf-8 -*-
'''
@File    :   sk_guassian_nb.py   
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 14:07   Deng      1.0         None
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import math


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]


class NaiveBayes:
    """
    高斯朴素贝叶斯
    """

    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(x_data):
        return sum(x_data) / float(len(x_data))

    # 标准差（方差）
    def stdev(self, x_data):
        avg = self.mean(x_data)
        return math.sqrt(sum([pow(x - avg, 2) for x in x_data]) / float(len(x_data)))

    # 概率密度函数
    def gaussian_probability(self, x_data, mean, stdev):
        exponent = math.exp(-(math.pow(x_data - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, x_data, y_data):
        labels = list(set(y_data))
        data = {label: [] for label in labels}
        for feature, label in zip(x_data, y_data):
            data[label].append(feature)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, x_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(x_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, x_test, y_test):
        """
        得分

        Args:
            x_test (): 训练集X
            y_test (): 训练集Y

        Returns:

        """
        right = 0
        for x_data, y_data in zip(x_test, y_test):
            label = self.predict(x_data)
            if label == y_data:
                right += 1

        return right / float(len(x_test))


if __name__ == '__main__':
    x_dat, y_dat = create_data()
    x_tr, x_tes, y_tr, y_tes = train_test_split(x_dat, y_dat, test_size=0.3)
    model = NaiveBayes()
    model.fit(x_tr, y_tr)
    print(model.predict([4.4, 3.2, 1.3, 0.2]))
    print(model.score(x_tes, y_tes))

    clf = GaussianNB()
    clf.fit(x_tr, y_tr)
    print(clf.score(x_tes, y_tes))
    print(clf.predict([[4.4, 3.2, 1.3, 0.2]]))
