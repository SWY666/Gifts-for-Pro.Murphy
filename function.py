from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt

#transfer method mainly for 9 layer in GR
def ffff(x):
    ranges = [(-10, -0.6),(-0.6, 0.3),(-0.3, 0), (0, 0.3), (0.3, 1), (1, 2), (2, 10)]
    scores = [-40, -10, -3, 3, 10, 25, 40]
    # scores = [-1, -1, -1, 1, 1, 1, 1]
    result = 0
    for i in range(len(ranges)):
        if ranges[i][0] < x <= ranges[i][1]:
            result = scores[i]

    return result
#use ffff on all items in data sets
def transfor_matrix(Z):
    if len(Z.shape) == 3:
        Z = Z[-1, :, :].copy()
    else:
        Z = Z[:, :].copy()
    news = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            news[i, j] = ffff(Z[i, j])
    return news
#divide matrix into different blocks using recursion method(Clustering)
def arrange_matrix_ut(datasets, shoot1):
    # print(len(datasets))
    # print(len(shoot1))
    counts = [i for i in range(len(shoot1))]
    db = AgglomerativeClustering()
    db.fit(datasets)
    lables = db.labels_
    dirs = {}
    for i in counts:
        if lables[i] not in dirs:
            dirs[lables[i]] = [shoot1[i][1]]
        else:
            dirs[lables[i]].append(shoot1[i][1])

    for i in dirs:
        tmp = dirs[i].copy()
        dirs[i] = [[r, tmp[r]] for r in range(len(tmp))]
        if len(dirs[i]) >= 2:
            # print("检查问题", dirs[i])
            current_datasets = [datasets[j[0]] for j in dirs[i]] #我要保持实际序号
            # dirs[i].sort(key=lambda x: np.sum(np.array(current_datasets[x[0]])))
            dirs[i] = arrange_matrix_ut(current_datasets, dirs[i])

    result = []
    for i in dirs:
        tmp = []
        for j in dirs[i]:
            tmp.append(j)
        result += tmp
    return result
#Clustering method
def change_matrix(matrix_in):
    if len(matrix_in.shape) == 3:
        target = matrix_in[-1, :, :].copy()
    else:
        target = matrix_in[:, :].copy()
    target = transfor_matrix(target)
    dlist = [target[:, i] for i in range(target.shape[1])]
    shoot1 = [[j, j] for j in range(target.shape[1])]
    dirs = arrange_matrix_ut(dlist, shoot1)
    dlist = [d[1] for d in dirs]
    # dlist = [current_Z[-1, :, :][:, d[1]] for d in dirs]

    dlist_ = [target[i, :] for i in range(target.shape[0])]
    shoot1 = [[j, j] for j in range(target.shape[0])]
    dirs = arrange_matrix_ut(dlist_, shoot1)
    dlist_ = [d[1] for d in dirs]

    return dlist_, dlist

def arrange_matrix_ut_v(datasets, shoot1):
    # print(len(datasets))
    # print(len(shoot1))
    counts = [i for i in range(len(shoot1))]
    db = AgglomerativeClustering(metric='precomputed')
    db.fit(cal_likelihood(datasets))
    lables = db.labels_
    dirs = {}
    for i in counts:
        if lables[i] not in dirs:
            dirs[lables[i]] = [shoot1[i][1]]
        else:
            dirs[lables[i]].append(shoot1[i][1])

    for i in dirs:
        tmp = dirs[i].copy()
        dirs[i] = [[r, tmp[r]] for r in range(len(tmp))]
        if len(dirs[i]) >= 2:
            print("检查问题", dirs[i])
            current_datasets = [datasets[j[0]] for j in dirs[i]] #我要保持实际序号
            dirs[i] = arrange_matrix_ut(current_datasets, dirs[i])

    result = []
    for i in dirs:
        tmp = []
        for j in dirs[i]:
            tmp.append(j)
        result += tmp
    return result

def arrange_matrixs(matrix_in):
    db = AgglomerativeClustering()
    datasets = [matrix_in[-1, :, i] for i in range(matrix_in.shape[2])]
    db.fit(datasets)
    lables = db.labels_
    shoot1 = [j for j in range(len(lables))]
    dirs = {}
    for i in range(len(shoot1)):
        if lables[i] not in dirs:
            dirs[lables[i]] = [i]
        else:
            dirs[lables[i]].append(i)

    results = []
    for i in dirs:
        tmp = []
        for j in dirs[i]:
            tmp.append(matrix_in[-1, :, j])
        results += tmp

    results = np.array(results)
    datasets = [results[i, :] for i in range(matrix_in.shape[1])]
    db.fit(datasets)
    lables = db.labels_
    shoot1 = [j for j in range(len(lables))]
    dirs = {}
    for i in range(len(shoot1)):
        if lables[i] not in dirs:
            dirs[lables[i]] = [i]
        else:
            dirs[lables[i]].append(i)

    results1 = []
    for i in dirs:
        for j in dirs[i]:
            results1.append(results[:, j])

    results1 = np.array(results1)
    return results1

def cal_likelihood(datasets):
    result = np.zeros((len(datasets), len(datasets)))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.linalg.norm(np.array(datasets[i]) - np.array(datasets[j]))

    return result

def arrange_matrix_utm(datasets, shoot1):
    # print("wolai")
    # print(len(datasets))
    # print(len(shoot1))
    counts = [i for i in range(len(shoot1))]
    db = DBSCAN(metric='precomputed')
    db.fit(cal_likelihood(datasets))
    lables = db.labels_
    dirs = {}
    for i in counts:
        if lables[i] not in dirs:
            dirs[lables[i]] = [shoot1[i][1]]
        else:
            dirs[lables[i]].append(shoot1[i][1])

    for i in dirs:
        tmp = dirs[i].copy()
        dirs[i] = [[r, tmp[r]] for r in range(len(tmp))]
        if len(dirs[i]) >= 2:
            print("检查问题", dirs[i])
            current_datasets = [datasets[j[0]] for j in dirs[i]] #我要保持实际序号
            # dirs[i].sort(key=lambda x: np.sum(np.array(current_datasets[x[0]])))
            dirs[i] = arrange_matrix_utm(current_datasets, dirs[i])

    result = []
    for i in dirs:
        tmp = []
        for j in dirs[i]:
            tmp.append(j)
        result += tmp
    return result

def change_matrixs(target):
    dlist = [target[:, i] for i in range(target.shape[1])]
    shoot1 = [[j, j] for j in range(target.shape[1])]
    dirs = arrange_matrix_ut(dlist, shoot1)
    dlist = [d[1] for d in dirs]
    # dlist = [current_Z[-1, :, :][:, d[1]] for d in dirs]

    dlist_ = [target[i, :] for i in range(target.shape[0])]
    shoot1 = [[j, j] for j in range(target.shape[0])]
    dirs = arrange_matrix_ut(dlist_, shoot1)
    dlist_ = [d[1] for d in dirs]

    return dlist_, dlist

def histo(matrix_in):
    item_list = matrix_in.flatten().copy()
    plt.figure()
    plt.hist(item_list, bins=100)
    plt.show()