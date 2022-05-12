# -*- encoding: utf-8 -*-
'''
@File    :   origin_train.py
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/12 11:06   Deng      1.0         None
'''

train = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]

w = [0, 0]
b = 0
l_rate = 1


# 使用梯度下降法更新权重
def update(data):
    global w, b
    w[0] = w[0] + l_rate * data[1] * data[0][0]
    w[1] = w[1] + l_rate * data[1] * data[0][1]
    b = b + l_rate * data[1]
    print(w, b)


# 计算到超平面的距离
def cal(data):
    global w, b
    res = 0
    for i in range(len(data[0])):
        res += data[0][i] * w[i]
    res += b
    res *= data[1]
    return res


# 检查是否可以正确分类
def check():
    flag = False
    for data in train:
        if cal(data) <= 0:
            flag = True
            update(data)
    if not flag:
        print("The result: w: " + str(w) + ", b: " + str(b))
        return False
    flag = False


if __name__ == '__main__':

    for i in range(3):
        check()
        if check() == False:
            print("train times:" + str(i + 1))
            break
