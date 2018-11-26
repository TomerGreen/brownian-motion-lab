import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from data_analyzer import *
import trackpy as tp
import pims
import os.path
import statsmodels.formula.api as smf
import theoretical_model
import logging
logging.basicConfig(level=logging.DEBUG)


# ============= EXPERIMENT CONFIGURATION CONSTANTS ============= #

TEMP_ABS_ERROR = 1      # The absolute temperature error. set to 1 degree.
DEFAULT_TEMPERATURE = 296     #23 deg. celcius
DEFAULT_VISCOSITY = 0.00093505      #100% water at 23 deg celcius
DEFAULT_VISCOSITY_ERROR = 0.00

# If a certain value does not appear, its default value will be used.
ENVIRONMENT_VARIABLES = {
    '1.csv':
        {
            'visc': 0.05,
            'visc_error': 0.005,
            'temp': 300
        },
    '2.csv':
        {
            'visc': 0.1,
        },
    '3.csv':
        {
            'temp': 300
        },
    '100%water.csv':
        {
        },
}

# ============= FUNCTIONS ============= #
RAW_DATA_PATH ='D:\\GDrive\\Lab II\\Raw Data\\week 2/100%water.csv'
TABLE2_PATH = '100%water.table2.csv'
TABLE3_PATH = '100%water.table3v2.csv'

def get_radius_error(radius):
    """
    takes into account z axis errors
    :param radius:
    :return:
    """
    return 0.1*radius

def get_chi_sq(table3_path):
    """
    :param table3_path: DataFrame with columns:
    :return: chi_squared
    """
    df = pd.read_csv(table3_path)

    sum = 0
    for i in range(df.particle.size):
        sum += np.square((df.coef[i] - df.theory_val[i]) / df.theory_err[i])
    return sum


def append_table3(data, particle, table3_path, particle_size=0):
    """
    :param data: dataframe
    :param particle: int
    :param table3_path: string
    :param particle_size: int
    :return:
    """
    particle_size = data[data['particle'] == 22][data['frame'] == 0]['size']
    radius = particle_size*theoretical_model.PIXEL_LENGTH_IN_METERS
    radius_err = get_radius_error(radius)

    part_sum = get_particle_sq_distance_data(data[data.particle == particle])
    # write data to table_3
    c, s, std_err = get_regression_table2(part_sum)

    theory_val, theory_err = theoretical_model.get_estimated_inverse_slope(
        data.temp[0], data.temp_error[0], data.visc[0], data.visc_error[0], radius, radius_err)

    df = pd.DataFrame([particle, c, s, radius,radius_err, std_err, theory_err, theory_val],
                      columns=['particle', 'coef', 'score', 'radius','radius_err', 'std_err', 'theory_err', 'theory_val'])
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
    :param part_sum: dataframe with r_sq, time_gap columns
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

    mod = smf.ols(formula='r_sq ~ time_gap - 1', data=part_sum)
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


def add_environment_variables(data, filepath):
    """
    Fills a data file with temperature and viscosity data, according to a dict at the top of this file.
    The name of the data file (for example, temp30.csv) must be given.
    """
    filename = os.path.basename(filepath)
    file_dict = ENVIRONMENT_VARIABLES[filename]
    if 'temp' in file_dict.keys():
        temp = file_dict['temp']
    else:
        temp = DEFAULT_TEMPERATURE
    if 'visc' in file_dict.keys():
        visc = file_dict['visc']
    else:
        visc = DEFAULT_VISCOSITY
    temp_error = TEMP_ABS_ERROR
    if 'visc_error' in file_dict.keys():
        visc_error = file_dict['visc_error']
    else:
        visc_error = DEFAULT_VISCOSITY_ERROR
    data['temp'] = temp
    data['visc'] = visc
    data['temp_error'] = temp_error
    data['visc_error'] = visc_error
    return data


def get_data(raw_data_path):
    """
    :param raw_data_path: string
    :return:
    """
    data = pd.read_csv(raw_data_path)
    data = tp.link_df(data, MAX_PIXELS_BW_FRAMES, memory=TRACKING_MEMORY)
    data = tp.filter_stubs(data, MIN_TRACK_LENGTH)
    data = cancel_avg_velocity_drift(data)
    data = add_environment_variables(data, raw_data_path)
    return data
