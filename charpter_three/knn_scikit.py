# -*- encoding: utf-8 -*-
'''
@File    :   knn_scikit.py
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/13 14:07   Deng      1.0         None
'''

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def generate_data():
    # 生成数据
    """
    代码中，生成60个训练样本，这60个样本分布在以centers参数指定中心点周围。cluster_std是标准差，用来指明生成的点分布的松散程度。
    生成的训练数据集放在变量X里面，数据集的类别标记放在y里面。
    make_blobs函数是为聚类产生数据集
    产生一个数据集和相应的标签
    n_samples:表示数据样本点个数,默认值100
    n_features:表示数据的维度，默认值是2
    centers:产生数据的中心点，默认值3
    cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
    center_box：中心确定之后的数据边界，默认值(-10.0, 10.0)
    shuffle ：洗乱，默认值是True
    random_state:官网解释是随机生成器的种子
    """
    centers = [[-2, 2], [2, 2], [0, 4]]
    data_x, data_y = make_blobs(n_samples=100, centers=centers, random_state=6, cluster_std=0.60)
    print(data_x)  # 坐标点
    print(data_y)  # 类别

    # 画出数据 这些点的分布情况在坐标轴上一目了然，其中三角形的点即各个类别的中心节点。
    plt.figure(figsize=(16, 10), dpi=72)
    data_center = np.array(centers)
    # 画出样本
    plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, s=100, cmap='cool')
    # 画出中心点
    plt.scatter(data_center[:, 0], data_center[:, 1], s=100, marker='^', c='orange')
    plt.savefig('knn_centers.png')
    plt.show()
    return data_x, data_y, centers


def knn_train(x_train, y_train, k):
    """
    模型训练

    Args:
        x_train ():
        y_train ():
        k ():

    Returns:

    """
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)
    return clf


def predict(clf, x_sample, x_train, y_train, centers):
    """
    对一个新样本进行预测：
    我们要预测的样本是[0, 2]，使用kneighbors()方法，把这个样本周围距离最*的5个点取出来。
    取出来的点是训练样本X里的索引，从0开始计算。
    """
    y_sample = clf.predict(x_sample)
    neighbors = clf.kneighbors(x_sample, return_distance=False)

    # 把待预测的样本以及和其最*的5个样本在图上标记出来。
    # 画出示意图
    plt.figure(figsize=(16, 10), dpi=72)
    c_data = np.array(centers)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=100, cmap='cool')  # 画出样本
    plt.scatter(c_data[:, 0], c_data[:, 1], s=100, marker='^', c='k')  # 中心点
    plt.scatter(x_sample[0][0], x_sample[0][1], marker="x",
                s=100, cmap='cool')  # 待预测的点
    for i in neighbors[0]:
        plt.plot([x_train[i][0], x_sample[0][0]], [x_train[i][1], x_sample[0][1]],
                 'k--', linewidth=0.6)  # 预测点与距离最*的5个样本的连线
    plt.savefig('knn_predict.png')
    plt.show()
    print(str(y_sample))


if __name__ == '__main__':
    x_tr, y_tr, c_tr = generate_data()
    model = knn_train(x_tr, y_tr, 5)
    x_new = np.array([[0, 2]])
    predict(model, x_new, x_tr, y_tr, c_tr)
