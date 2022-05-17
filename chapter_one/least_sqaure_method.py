# -*- encoding: utf-8 -*-
'''
@File    :   least_sqaure_method.py
@License :   (C)Copyright 2022-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/11 15:16   Deng      1.0         最小二乘法拟合曲线

说明：最小二乘法拟合曲线（由来：高斯于1823年在误差e1 ,… , en独立同分布的假定下,证明了最小二乘方法的一个最优性质: 在所有无偏的线性估计类中,最小二乘方法是其中方差最小的！）
举例：我们用目标函数y=sin2πx, 加上一个正太分布的噪音干扰，用多项式去拟合
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq


def real_func(x_input):
    """
    目标函数

    Args:
        x_input (): 入参

    Returns:

    """
    return np.sin(2 * np.pi * x_input)


def fit_func(weight, x_input):
    """
    拟合函数：多项式

    Args:
        weight (): 权重
        x_input (): x

    Returns:

    """
    # 生成多项式
    func = np.poly1d(weight)
    return func(x_input)


def residuals_func(weight, x_input, y_input):
    """
    残差

    Args:
        weight ():
        x_input ():
        y_input ():

    Returns:

    """
    return fit_func(weight, x_input) - y_input


def fitting(x_noisz, y_noisz, x_in, pic_name, power_set=0):
    """
    拟合函数

    Args:
        x_noisz ():
        y_noisz ():
        x_in ():
        y_in ():
        pic_name ():
        power_set ():

    Returns:

    """
    # 随机初始化多项式参数
    p_init = np.random.rand(power_set + 1)
    print('p_init:', p_init)

    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x_noisz, y_noisz))
    print('Fitting Parameters:', p_lsq[0])
    # 可视化
    # initialize plot parameters
    show_image(p_lsq, x_noisz, y_noisz, x_in, pic_name)
    return p_lsq


def residuals_func_regularization(weight, x_in, y_in, regularization=0.0001):
    """
    正则化

    Args:
        weight ():
        x_in ():
        y_in ():

    Returns:

    """
    # 残差
    ret = residuals_func(weight, x_in, y_in)
    # L2范数作为正则化项
    return np.append(ret, np.sqrt(0.5 * regularization * np.square(weight)))


def show_image(p_lsq, x_noise, y_noise, x_in, pic_name):
    """
    展示图像

    Args:
        p_lsq ():
        x_noise ():
        y_noise ():
        x_in ():
        y_in ():
        pic_name ():

    Returns:

    """
    plt.rcParams['figure.figsize'] = (10 * 16 / 9, 10)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08)

    # 画图
    print('picture name: %s, len of data: %d' % (pic_name, x_in.size))
    plt.plot(x_in, real_func(x_in), label='real')
    plt.plot(x_in, fit_func(p_lsq[0], x_in), label='fitted curve')
    plt.plot(x_noise, y_noise, 'bo', label='noise')
    plt.title(pic_name)
    plt.legend()
    plt.show()


# 数据
# 十个点
x_noise = np.linspace(0, 1, 10)
x_ = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x_noise)
y_noise = [np.random.normal(0, 0.1) + y1 for y1 in y_]

if __name__ == '__main__':
    # power_set不同时候
    p_lsq_0 = fitting(x_noise, y_noise, x_, "power_set 0", 0)
    p_lsq_1 = fitting(x_noise, y_noise, x_, "power_set 1", 1)
    p_lsq_3 = fitting(x_noise, y_noise, x_, "power_set 3", 3)
    p_lsq_6 = fitting(x_noise, y_noise, x_, "power_set 6", 6)
    p_lsq_9 = fitting(x_noise, y_noise, x_, "power_set 9", 9)

    # 当power_set为9的时候过拟合, 需要正则化

    # 最小二乘法,加正则化项
    p_init = np.random.rand(9 + 1)
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x_noise, y_noise))
    plt.plot(x_, real_func(x_), label='real')
    plt.plot(x_, fit_func(p_lsq_9[0], x_), label='fitted curve')
    plt.plot(x_, fit_func(p_lsq_regularization[0], x_), label='regularization')
    plt.plot(x_noise, y_noise, 'bo', label='noise')
    plt.legend()
    plt.show()
