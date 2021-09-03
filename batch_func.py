import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from function import *
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

models = AgglomerativeClustering()

x = np.linspace(1, 35, 35)
y = np.linspace(1, 34, 34)
X, Y = np.meshgrid(x, y)

#返回每一个点和其他点的矩阵
def return_distance(point_list):
    lens = len(point_list)
    result = np.zeros(lens, lens)
    for i in range(lens):
        for j in range(lens):
            result[i, j] = mean_squared_error(point_list[i], point_list[j])

    total_len = [np.sum(result[i, :]) for i in range(lens)]

    return total_len



def accusition_sampleb(model, remaining_point_sets, threshold=50, shape=(34, 35)):
    expp, stds = model.predict(remaining_point_sets, return_std=True)
    sttd = stds.copy()
    sttd = sorted(sttd, reverse=True)
    print("sttd", sttd)
    # print("最大", np.max(stds))
    # print(len(stds))
    expp_final = np.zeros(shape)
    stds_final = np.zeros(shape)
    for index in range(len(stds)):
        print(index, stds[index], remaining_point_sets)
        expp_final[int(remaining_point_sets[index][2]) - 1, int(remaining_point_sets[index][1]) - 1] = expp[index]
        stds_final[int(remaining_point_sets[index][2]) - 1, int(remaining_point_sets[index][1]) - 1] = stds[index]
    stds_judge = np.c_[stds, [x for x in range(len(remaining_point_sets))], [x for x in remaining_point_sets]].tolist()
    stds_judge = sorted(stds_judge, key=lambda x: x[0])
    stds_judges = stds_judge.copy()
    stds_judge = stds_judge[-threshold:]

    # if threshold >= 30:
    #     std_cluster = stds_judges.copy()
    #     print(std_cluster)
    #     print(np.array(std_cluster))
    #     # models.fit(std_cluster)
    #     print(models.fit_predict(std_cluster))

    # print(stds_judge)
    # the_choosen_ones = [[int(stds_judge[x][1]), int(stds_judge[x][2])] for x in range(len(stds_judge))]
    the_choosen_orders = [int(stds_judge[x][1]) for x in range(len(stds_judge))]
    # print(the_choosen_orders)
    # print(remaining_point_sets[the_choosen_orders[0]])
    plt.figure(151)

    plt.contourf(X, Y, stds_final)
    plt.title("see this")
    # sns.heatmap(stds_final, annot=True)
    plt.colorbar()
    plt.scatter([int(remaining_point_sets[item][1]) for item in the_choosen_orders],
                [int(remaining_point_sets[item][2]) for item in the_choosen_orders],
                c="red", marker="x")
    plt.figure(152)

    plt.contourf(X, Y, expp_final)
    plt.title("std matrix")
    # sns.heatmap(stds_final, annot=True)
    # plt.colorbar()
    plt.scatter([int(remaining_point_sets[item][1]) for item in the_choosen_orders],
                [int(remaining_point_sets[item][2]) for item in the_choosen_orders],
                c="red", marker="x")
    plt.show()
    return stds, the_choosen_orders