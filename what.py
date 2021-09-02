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

def order_cal(TOTAL_TIMES, POINT_SET, input_1, Z_):
    MSE_orderss = []
    rbf = ConstantKernel(2.5) + RBF(length_scale=0.1)
    reg = GaussianProcessRegressor(rbf)
    for i in range(TOTAL_TIMES):
        orders = [i for i in range(len(Z_))]
        random.shuffle(orders)
        orders = orders[:POINT_SET]
        inputs = []
        result = []
        for index in orders:
            inputs.append(input_1[index])
            result.append(Z_[index])

        result = np.array(result)
        result = result.reshape(-1, 1)
        inputs = np.array(inputs).reshape(-1, 1)
        reg.fit(inputs, result)
        output = reg.predict(np.array(input_1[:]))
        MSE_orderss.append(mean_squared_error(output, Z_))
    # plt.figure(2)
    # plt.plot(x, Z_, "red", label='real')
    # plt.plot(x, output, "blue", label='yaht')
    # plt.scatter(inputs, result)
    # plt.title("sorted 1D function")
    # print(len())
    # plt.show()
    MSE_means = np.mean(np.array(MSE_orderss))
    return MSE_means


if __name__ == "__main__":
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import random

    TOTAL_TIMES = 20
    POINT_SET = 15
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    # Z = f(x)

    Z = [1 for i in range(10)] + [2 for i in range(10)] + [3 for i in range(10)] + [4 for i in range(10)] + [5 for i in range(10)]
    #########################################################################################################
    MSE_disorder = []
    random.shuffle(Z)
    Z = np.array(Z)

    input_1 = []
    Z_ = []
    for i in range(len(x)):
        input_1.append([x[i]])
        Z_.append(Z[i])
    disorder_MSE = []
    for point_set in range(15, 50, 1):
        disorder_MSE.append(order_cal(TOTAL_TIMES, point_set, input_1, Z_))



    ####################################
    TOTAL_TIMES = 20
    POINT_SET = 15
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    # Z = f(x)

    Z = [1 for i in range(10)] + [2 for i in range(10)] + [3 for i in range(10)] + [4 for i in range(10)] + [5 for i in
                                                                                                             range(10)]
    # random.shuffle(Z)
    Z = np.array(Z)

    order_MSE = []
    input_1 = []
    Z_ = []
    for i in range(len(x)):
        input_1.append([x[i]])
        Z_.append(Z[i])

    order_MSE = []
    for point_set in range(15, 50, 1):
        order_MSE.append(order_cal(TOTAL_TIMES, point_set, input_1, Z_))

    plt.figure(3)
    plt.plot([x for x in range(15, 50, 1)], order_MSE, "red")
    plt.plot([x for x in range(15, 50, 1)], disorder_MSE, "blue")
    plt.legend(["MSE of order ones","MSE of disorder ones"])
    plt.xlabel("the turns of trial")
    plt.ylabel("MSE")
    plt.show()