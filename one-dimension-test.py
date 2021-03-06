import math
import numpy as np
from matplotlib import pyplot as plt
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

def plot_unit_gaussian_samples(D):
    p = plt.figure(plot_width=800, plot_height=500,
              title='Samples from a unit {}D Gaussian'.format(D))

    xs = np.linspace(0, 1, D)
    for i in range(10):
           ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
           p.line(xs, ys, line_width=1)

    return p

def f(X):
    return 1 + X**2

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    import random
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    # Z = f(x)

    Z = [1 for i in range(10)] + [2 for i in range(10)] + [3 for i in range(10)] + [4 for i in range(10)] + [5 for i in range(10)]
    # random.shuffle(Z)
    Z = np.array(Z)

    input_1 = []
    Z_ = []
    for i in range(len(x)):
        input_1.append([x[i]])
        Z_.append(Z[i])

    orders = [i for i in range(len(x))]
    random.shuffle(orders)
    orders = orders[:20]
    inputs = []
    result = []
    for index in orders:
        inputs.append(input_1[index])
        result.append(Z_[index])

    result = np.array(result)
    result = result.reshape(-1, 1)
    inputs = np.array(inputs)
    # 创建高斯过程回归,并训练
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    # kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
    # reg = RandomForestRegressor(n_estimators=30)
    # rbf = RBF(length_scale=1.0)
    rbf = ConstantKernel(2.5) + RBF(length_scale=0.1)
    reg = GaussianProcessRegressor(rbf)
    reg.fit(inputs, result)
    output = reg.predict(np.array(input_1[:]))
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    plt.figure()
    plt.plot(x, Z, "red", label='real')
    # plt.plot(x, output, "blue", label='yaht')
    # plt.scatter(inputs, result)
    plt.title("sorted 1D function")
    # plt.savefig(f'./pic_output/layer{9}.eps', dpi=600, format='eps')

    plt.savefig(f'./pic_output/layer{9}.svg', format="svg")
    plt.show()
    # fig = plt.figure()
    ##

    ##
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, out_new, 50)
    # ax.contour3D(X, Y, Z, 50)
    # plt.show()