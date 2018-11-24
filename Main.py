import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


def get_regression(part_sum):
    """
    per particle:
    creates linear regression for r_sq vs t,
    plots graph
    returns coeff and r^2 score
    :param part_sum: dataframe with r_sq, frame_gap columns
    :return: coeff, score
    """
    x = part_sum.frame_gap
    x = x[:, None]
    y = part_sum.r_sq
    y = y[:, None]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_fit = regr.predict(x)
    plt.scatter(x, y)
    plt.plot(x, y_fit)
    plt.show()
    return regr.coef_[0], regr.score(x, y)

def append_particle_to_csv(part_data,sum_file):


