import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from data_analyzer import *
import trackpy as tp
import pims
import os.path


# ============= EXPERIMENT CONFIGURATION CONSTANTS ============= #

TEMP_ABS_ERROR = 1      # The absolute temperature error. set to 1 degree.
DEFAULT_TEMPERATURE = 300
DEFAULT_VISCOSITY = 0.05
DEFAULT_VISCOSITY_ERROR = 0.005

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
    '4.csv':
        {
        },
}

# ============= FUNCTIONS ============= #


RAW_DATA_PATH ='D:\\GDrive\\Lab II\\Raw Data\\week 2/100%water.csv'
TABLE2_PATH = '100%water2.table2.csv'
TABLE3_PATH = '100%water2.table3.csv'


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
    returns coeff and r^2 score
    :param part_sum: dataframe with r_sq, time_gap columns
    :return: coeff: array size 1, score: int
    """
    x = part_sum.time_gap
    x = x[:, None]
    y = part_sum.r_sq
    y = y[:, None]
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x, y)
    y_fit = regr.predict(x)
    #plt.scatter(x, y)
    #plt.plot(x, y_fit)
    #plt.show()
    return regr.coef_[0], regr.score(x, y)


def fit_table3(table3_path):
    """
    fit and plot 2KT/(3pi*eta*a) from table2 fit vs length of particle
    :param table3_path: string
    :return:
    """
    regr = linear_model.LinearRegression(False)
    t = pd.read_csv(table3_path)
    x = t.length
    x = x[:,None]
    y = t.coef
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
    c, s = get_regression_table2(part_sum)
    c = c[0]
    df = pd.DataFrame([[particle, c, s, particle_size]],
                      columns=['particle', 'coef', 'score', 'size'])
    # sum_file = '100%water.table3.csv'
    with open(table3_path, 'a') as f:
        df.to_csv(f, header=None)


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
