import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from data_analyzer import *
import trackpy as tp
import pims
import statsmodels.formula.api as smf
import theoretical_model
import logging
logging.basicConfig(level=logging.DEBUG)

RAW_DATA_PATH ='D:\\GDrive\\Lab II\\Raw Data\\week 2/100%water.csv'
TABLE2_PATH = '100%water2.table2.csv'
TABLE3_PATH = '100%water2.table3.csv'


def get_chi_sq(table3_path):
    """

    :param table3_path: DataFrame with columns:
    :return: chi_squared
    """

    df = pd.read_csv(table3_path)

    raise Exception('get_chi_sq not implemented yet')

def append_table3(data, particle, table3_path, particle_size):
    """
    :param data: dataframe
    :param particle: int
    :param table3_path: string
    :param particle_size: int
    :return:
    """
    part_sum = get_particle_sq_distance_data(data[data.particle == particle])
    # write data to table_3
    c, s, std_err = get_regression_table2(part_sum)

    theory_val, theory_err = theoretical_model.get_estimated_slope()

    df = pd.DataFrame([[particle, c, s, particle_size, std_err, theory_err,theory_val]],
                      columns=['particle', 'coef', 'score', 'length', 'std_err','theory_err','theory_val'])
    # sum_file = '100%water.table3.csv'
    with open(table3_path, 'a') as f:
        df.to_csv(f, header=None)


def show_annotated_first_frame(data,img_dir_path):
    """
    :param data: dataframe
    :param img_dir_path: string
    :return:
    """
    frames = pims.ImageSequence('{}*.jpg'.format(img_dir_path))
    plt.figure()
    tp.annotate(data[data['frame'] == 0], frames[0])


def get_regression_table2(part_sum):
    """
    per particle:
    creates linear regression for r_sq vs t,
    plots graph
    returns coeff and r^2 score and stderr
    :param part_sum: dataframe with r_sq, frame_gap columns
    :return: coeff: array size 1, score: int
    """
    # x = part_sum.frame_gap
    # x = x[:, None]
    # y = part_sum.r_sq
    # y = y[:, None]
    # regr = linear_model.LinearRegression(fit_intercept=False)
    # regr.fit(x, y)
    # y_fit = regr.predict(x)
    # plt.scatter(x, y)
    # plt.plot(x, y_fit)
    # plt.show()
    # stderr = 0

    mod = smf.ols(formula='r_sq ~ frame_gap - 1', data=part_sum)
    res = mod.fit()
    return res.params[0], res.rsquared, res.bse[0]
    # return regr.coef_[0], regr.score(x, y), stderr



def fit_table3(table3_path):
    """
    fit and plot 2KT/(3pi*eta*a) from table2 fit vs length of particle
    :param table3_path: string
    :return:
    """
    regr = linear_model.LinearRegression(False)
    t = pd.read_csv(table3_path)
    x = t.length.values
    x = x[:,None]
    y = t.coef.values
    y = y[:,None]
    y = [1/i for i in y]

    regr.fit(x,y)
    y_fit = regr.predict(x)
    plt.plot(x,y_fit),plt.scatter(x,y),plt.show()


def append_table2(data, particle, table2_path):
    """

    :param data: dataframe
    :param particle: int
    :param table2_path: string
    :return:
    """
    part_sum = get_particle_sq_distance_data(data[data.particle == particle])
    # write data to table_2
    part_sum['particle'] = particle
    with open(table2_path, 'a') as f:
        part_sum.to_csv(f, header=None)





def main(data, particle, table2_path, table3_path):
    """
    update table2 and table 3 per particle
    :param data: DataFrame
    :param particle: int
    :param table2_path: string
    :param table3_path: string
    :return:
    """
    append_table2(data, particle, table2_path)
    append_table3(data, particle, table3_path)


def get_data(raw_data_path):
    """
    :param raw_data_path: string
    :return:
    """
    data = pd.read_csv(raw_data_path)
    data = tp.link_df(data, MAX_PIXELS_BW_FRAMES, memory=TRACKING_MEMORY)
    data = tp.filter_stubs(data, MIN_TRACK_LENGTH)
    data = cancel_avg_velocity_drift(data)
    return data
