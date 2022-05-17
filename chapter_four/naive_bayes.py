# -*- encoding: utf-8 -*-
'''
@File    :   naive_bayes.py
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 13:45   Deng      1.0         None
'''

import numpy as np


def train(x_train, y_train, feature):
    """
    模型训练

    Args:
        x_train (): 训练数据X
        y_train (): 训练数据Y
        feature (): 特征矩阵

    Returns:

    """
    global class_num, label
    class_num = 2  # 分类数目
    label = [1, -1]  # 分类标签
    feature_len = len(feature)  # 特征长度

    prior_prob = np.zeros(class_num)  # 初始化先验概率
    con_prob = np.zeros((class_num, feature_len, 2))  # 初始化条件概率

    positive_count = 0  # 统计正类
    negative_count = 0  # 统计负类

    for i in range(len(y_train)):
        if y_train[i] == 1:
            positive_count += 1
        else:
            negative_count += 1

    prior_prob[0] = positive_count / len(y_train)  # 求得正类的先验概率
    prior_prob[1] = negative_count / len(y_train)  # 求得负类的先验概率

    '''
    con_prob是一个2*3*2的三维列表，第一维是类别分类，第二维和第三维是一个3*2的特征分类
    '''
    # 分为两个类别
    for i in range(class_num):
        # 对特征按行遍历
        for j in range(feature_len):
            # 遍历数据集，并依次做判断
            for k in range(len(y_train)):
                if y_train[k] == label[i]:  # 相同类别
                    if x_train[k][0] == feature[j][0]:
                        con_prob[i][j][0] += 1
                    if x_train[k][1] == feature[j][1]:
                        con_prob[i][j][1] += 1

    class_label_num = [positive_count, negative_count]  # 存放各类型的数目
    for i in range(class_num):
        for j in range(feature_len):
            con_prob[i][j][0] = con_prob[i][j][0] / class_label_num[i]  # 求得i类j行第一个特征的条件概率
            con_prob[i][j][1] = con_prob[i][j][1] / class_label_num[i]  # 求得i类j行第二个特征的条件概率

    return prior_prob, con_prob

# 给定数据进行分类
def predict(testset, prior_prob, con_prob, feature):
    """
    分类

    Args:
        testset ():
        prior_prob ():
        con_prob ():
        feature ():

    Returns:

    """
    result = np.zeros(len(label))
    for i in range(class_num):
        for j in range(len(feature)):
            if feature[j][0] == testset[0]:
                con_a = con_prob[i][j][0]
            if feature[j][1] == testset[1]:
                con_b = con_prob[i][j][1]
        result[i] = con_a * con_b * prior_prob[i]

    result = np.vstack([result, label])

    return result

if __name__ == '__main__':
    X_train = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
               [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
               [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
    Y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

    # 构造3×2的列表
    feature = [[1, 'S'],
               [2, 'M'],
               [3, 'L']]

    testset = [2, 'S']

    prior_prob, con_prob = train(X_train, Y_train, feature)

    result = predict(testset, prior_prob, con_prob, feature)
    print('The result:', result)
