# -*- encoding: utf-8 -*-
'''
@File    :   dual_train.py   
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/12 15:36   Deng      1.0         None
'''

import numpy as np

train = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])

a = np.array([0, 0, 0])
b = 0
l_rate = 1
Gram = np.array([])
y = np.array(range(len(train))).reshape(1, 3)  # 标签
x = np.array(range(len(train) * 2)).reshape(3, 2)  # 特征

# 计算Gram矩阵
def gram():
    g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(len(train)):
        for j in range(len(train)):
            g[i][j] = np.dot(train[i][0], train[j][0])
    return g

# 更新权重
def update(i):
    global a, b
    a[i] = a[i] + 1
    b = b + train[i][1]
    print(a, b)

# 计算到超平面的距离
def cal(key):
    global a, b, x, y
    i = 0
    for data in train:
        y[0][i] = data[1]
        i = i + 1
    temp = a * y
    res = np.dot(temp, Gram[key])
    res = (res + b) * train[key][1]
    return res[0]

# 检查是否可以正确分类
def check():
    global a, b, x, y
    flag = False
    for i in range(len(train)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        i = 0
        for data in train:
            y[0][i] = data[1]
            x[i] = data[0]
            i = i + 1
        temp = a * y
        w = np.dot(temp, x)
        print("The result: w: " + str(w) + ", b: "+ str(b))
        return False
    flag = False

if __name__ == '__main__':
    Gram = gram()  # 初始化Gram矩阵
    for i in range(1000):
        check()
        if check() == False:
            print("train times:" + str(i+1))
            break


