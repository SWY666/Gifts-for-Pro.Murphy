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
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import random

    TOTAL_TIMES = 100
    POINT_SET = 40
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    # Z = f(x)

    Z = [1 for i in range(10)] + [2 for i in range(10)] + [3 for i in range(10)] + [4 for i in range(10)] + [5 for i in range(10)]

    # random.shuffle(Z)
    Z = np.array(Z)

    MSE_order = []
    input_1 = []
    Z_ = []
    for i in range(len(x)):
        input_1.append([x[i]])
        Z_.append(Z[i])

    for i in range(TOTAL_TIMES):
        orders = [i for i in range(len(x))]
        random.shuffle(orders)
        orders = orders[:POINT_SET]
        inputs = []
        result = []
        for index in orders:
            inputs.append(input_1[index])
            result.append(Z_[index])

        result = np.array(result)
        result = result.reshape(-1, 1)
        inputss = np.array(inputs)
        inputs = np.array(inputs).reshape(-1, 1)
        # kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
        # reg = RandomForestRegressor(n_estimators=30)
        # rbf = RBF(length_scale=1.0)
        rbf = ConstantKernel(2.5) + RBF(length_scale=0.1)
        reg = GaussianProcessRegressor(rbf)
        reg.fit(inputs, result)
        output = reg.predict(np.array(input_1[:]))
        plt.figure(1)
        plt.plot(x, Z, "red", label='real')
        plt.title("sorted 1D function")
        plt.plot(x, output, "blue", label='yaht')
        print(inputss)
        plt.scatter(inputss, result)
        print("MSE_order", mean_squared_error(output, Z))
        MSE_order.append(mean_squared_error(output, Z))
        # plt.show()

    #########################################################################################################

    MSE_disorder = []
    random.shuffle(Z)
    for i in range(TOTAL_TIMES):
        Z = np.array(Z)

        input_1 = []
        Z_ = []
        for i in range(len(x)):
            input_1.append([x[i]])
            Z_.append(Z[i])

        orders = [i for i in range(len(x))]
        random.shuffle(orders)
        orders = orders[:POINT_SET]
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
        plt.figure(2)
        plt.plot(x, Z, "red", label='real')
        plt.plot(x, output, "blue", label='yaht')
        plt.scatter(inputs, result)
        plt.title("sorted 1D function")
        MSE_disorder.append(mean_squared_error(output, Z))

    plt.figure(3)
    plt.plot([x for x in range(len(MSE_order))], MSE_order, "red")
    plt.plot([x for x in range(len(MSE_order))], MSE_disorder, "blue")
    plt.legend(["MSE of order ones","MSE of disorder ones"])
    plt.xlabel("the turns of trial")
    plt.ylabel("MSE")
    plt.show()