# -*- encoding: utf-8 -*-
'''
@File    :   iris_data.py   
@License :   (C)Copyright 2022-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/23 15:21   Deng      1.0         None
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def create_data():
    """
    构造数据

    Returns: 鸢尾花数据

    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]
