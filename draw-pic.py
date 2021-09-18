if __name__ == "__main__":
    from main import get_ininital_datas_curve_lines, dataset_space_matrix


    # new sampling points
    def accusition_samplen(model, remaining_point_sets, inputss, threshold=50, shape=(34, 35)):
        _, stds = model.predict(remaining_point_sets, return_std=True)
        # stds_final = np.zeros(shape)
        # for index in range(len(stds)):
        #     stds_final[int(remaining_point_sets[index][0]), int(remaining_point_sets[index][1])] = stds[index]
        # expectations_final[int(remaining_point_sets[index][1]) - 1, int(remaining_point_sets[index][0]) - 1] = expectations[index]

        stds_judge = np.c_[
            stds, [x for x in range(len(remaining_point_sets))], [x for x in remaining_point_sets]].tolist()
        stds_judge = sorted(stds_judge, key=lambda x: x[0])
        stds_judge = stds_judge[-threshold:]
        # print(stds_judge)
        # the_choosen_ones = [[int(stds_judge[x][1]), int(stds_judge[x][2])] for x in range(len(stds_judge))]
        the_choosen_orders = [int(stds_judge[x][1]) for x in range(len(stds_judge))]
        # print(the_choosen_orders)
        # print(remaining_point_sets[the_choosen_orders[0]])
        # plt.figure()
        # plt.contourf(X, Y, stds_final)
        # plt.title("see this")
        # plt.figure()
        # plt.contourf(X, Y, expectations_final)
        # plt.title("see this expectataions")
        # plt.show()
        return stds, the_choosen_orders


    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from sklearn.ensemble import RandomForestRegressor
    from function import *

    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)

    my_starter = get_ininital_datas_curve_lines()
    datapool = dataset_space_matrix()
    # select layer
    start = 8
    end = 9
    X, Y, Z, x, y = datapool.show_one_layers(start, end)

    cmaps = [('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                 'gist_ncar'])]

    plt.figure(101)
    plt.contourf(X, Y, np.transpose(Z[0, :, :]), cmap=plt.get_cmap("binary"))
    # plt.scatter([item[1] for item in inputs],
    #             [item[2] for item in inputs],
    #             c="red", marker="x")
    # plt.colorbar()
    plt.title("predict")

    plt.savefig("./pic_output/test.svg", format="svg")
    plt.show()