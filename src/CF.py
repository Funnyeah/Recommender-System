import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data preprocessing
# Import data
user = pd.read_csv("users.txt", names = ['userid'])
user['id'] = range(len(user))
netflix_train = pd.read_csv("netflix_train.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_train = netflix_train.merge(user, left_on='user_id', right_on='userid')
netflix_test = pd.read_csv("netflix_test.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_test = netflix_test.merge(user, left_on='user_id', right_on='userid')

X_train = netflix_train.pivot(index='id', columns='film_id', values='rating')
X_test = netflix_test.pivot(index='id', columns='film_id', values='rating')
print(X_train.head())
print(X_test.head())


# Collaborate Filtering
# Compute the overall mean and mean by row and column
mu = np.mean(np.mean(X_train))
bx = np.array(np.mean(X_train, axis=1) - mu)
by = np.array(np.mean(X_train, axis=0) - mu)
# Compute the similarity matrix
X = X_train.sub(bx+mu, axis=0)   # Demean
X = X.div(np.sqrt(np.sum(np.square(X), axis=1)), axis=0)
X.fillna(0, inplace=True)
similarity_matrix = np.dot(X, X.T)
# Compute the point matrix using CF
X_train = np.array(X_train.fillna(0))
for i in range(X_train.shape[0]):
    indexs = np.argsort(similarity_matrix[i, :])[::-1]
    for j in range(X_train.shape[1]):
        if X_train[i, j] == 0:
            sum = 0
            num = 0
            simi = 0
            k = 0
            while num < 3 & k < X_train.shape[1]:    # top 3
                if X_train[indexs[k], j] > 0:
                    sum = sum + similarity_matrix[i, indexs[k]] * (X_train[indexs[k], j] - mu - bx[indexs[k]] - by[j])
                    simi = simi + similarity_matrix[i, indexs[k]]
                    k = k+1
                    num = num + 1
                else:
                    k = k+1
            if simi != 0:
                X_train[i, j] = mu + bx[i] + by[j] + sum/simi
            else:
                X_train[i, j] = mu + bx[i] + by[j]
        else:
            continue
sum = 0
for index, rows in netflix_test.iterrows():
    sum = sum + np.square(X_train[rows['id'], rows['film_id']-1] - rows['rating'])
    print(X_train[rows['id'], rows['film_id']-1])
    print(sum)
RMSE = np.sqrt(sum/netflix_test.shape[0])
print(RMSE)







###  注释

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data preprocessing
# Import data
user = pd.read_csv("users.txt", names = ['userid'])
user['id'] = range(len(user))
netflix_train = pd.read_csv("netflix_train.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_train = netflix_train.merge(user, left_on='user_id', right_on='userid')
netflix_test = pd.read_csv("netflix_test.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_test = netflix_test.merge(user, left_on='user_id', right_on='userid')

X_train = netflix_train.pivot(index='id', columns='film_id', values='rating')
X_test = netflix_test.pivot(index='id', columns='film_id', values='rating')
print(X_train.head())
print(X_test.head())


"""
1.首先计算了整个矩阵X_train的平均值mu，并分别计算了每个用户和每个商品的相对于整体均值mu的平均打分偏差（bx和by）。
2.然后对矩阵X_train进行了去均值操作（减去整体均值mu和对应行均值bx），得到了去均值后的矩阵X。
3.接着，对去均值后的矩阵X的每一行向量进行了标准化处理，使其成为单位向量，模长为1，以便计算余弦相似度。
4.由于标准化过程中可能会产生NaN值（除数为0），代码使用fillna(0)将NaN值替换为0，确保相似度矩阵计算的稳定性。
5.最后，通过计算矩阵X与其转置X.T的乘积得到相似度矩阵similarity_matrix，其中similarity_matrix[i][j]表示用户i和用户j之间的余弦相似度。
这个过程是一种协同过滤的方法，通过计算用户之间的余弦相似度来衡量它们之间的相似性，从而进行推荐或类似的任务。
在代码中，通过去均值和标准化处理，可以更好地捕捉用户之间打分的相对差异，从而提高相似度计算的准确性。
"""
# Collaborate Filtering
# Compute the overall mean and mean by row and column
mu = np.mean(np.mean(X_train))
bx = np.array(np.mean(X_train, axis=1) - mu)
by = np.array(np.mean(X_train, axis=0) - mu)
# Compute the similarity matrix
X = X_train.sub(bx+mu, axis=0)   # Demean
X = X.div(np.sqrt(np.sum(np.square(X), axis=1)), axis=0)
X.fillna(0, inplace=True)
similarity_matrix = np.dot(X, X.T)


# Compute the point matrix using CF
X_train = np.array(X_train.fillna(0))
for i in range(X_train.shape[0]): # 遍历用户i
    indexs = np.argsort(similarity_matrix[i, :])[::-1] # i的最相似用户index下标列表
    for j in range(X_train.shape[1]): # 遍历商品j
        if X_train[i, j] == 0: # 如果用户i对商品j无评分则协同打分
            sum = 0 # 最相似的若干人总分数之和
            num = 0 # 最相似人数
            simi = 0 # 最相似人的总相似度之和
            k = 0 # 指示变量，防止所有相似用户对商品j都没分数导致越界
            while num < 3 & k < X_train.shape[1]:    # top 3 & 防止越界(这里是用户和商品维度一致，bug没显现，正确的应该是k < X_train.shape[0])
                if X_train[indexs[k], j] > 0: # 最相似用户对商品j的有评分的话
                    # 总分数 = 用户i与前3个相似用户的相似度*（该3个相似用户k对商品j的评分-整体均值评分-用户偏移均值评分-商品偏移均值评分）
                    sum = sum + similarity_matrix[i, indexs[k]] * (X_train[indexs[k], j] - mu - bx[indexs[k]] - by[j])
                    # 总相似度 = 前3个相似用户的相似度之和
                    simi = simi + similarity_matrix[i, indexs[k]]
                    k = k+1
                    num = num + 1
                else: # 最相似用户对商品j的无评分即=0
                    k = k+1
            if simi != 0: # 相似度不为0，也就是说有用户对j有评分，那么基线分+3个相似用户均值评分
                X_train[i, j] = mu + bx[i] + by[j] + sum/simi
            else: # 相似度为0，也就是说没用户对j有评分，数据极度稀疏，采用baseline评分
                X_train[i, j] = mu + bx[i] + by[j]
        else:# 如果用户i对商品j有评分则继续找没评分的协同打分
            continue
sum = 0
for index, rows in netflix_test.iterrows():
    sum = sum + np.square(X_train[rows['id'], rows['film_id']-1] - rows['rating'])
    print(X_train[rows['id'], rows['film_id']-1])
    print(index,sum)
RMSE = np.sqrt(sum/netflix_test.shape[0])
print(RMSE)
