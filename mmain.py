import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
# from data_analyzer import *
import trackpy as tp
import pims
import os.path
import statsmodels.formula.api as smf
import theoretical_model
import logging
import data_analyzer as analyzer
logging.basicConfig(level=logging.DEBUG)
import pandas as pd

import scipy.constants as const
# ============= EXPERIMENT CONFIGURATION CONSTANTS ============= #

DEFAULT_ENV_VARIABLES = {
    'temp': 296,    #23 deg. celcius
    'temp_error': 1,
    'visc': 0.0035713,      #100% water at 23 deg celcius
    'visc_error': 0.0003559
}

CEL_TO_KEL = 273
# If a certain value does not appear, its default value will be used.
ENVIRONMENT_VARIABLES = {
    '95%glys_10%part90%sol2.csv':
        {
            'visc': 0.138,
            'visc_error': 0.11,
        },
    '95%glys_10%part90%sol1.csv':
        {
            'visc': 0.138,
            'visc_error': 0.11,
        },
    '85%glys_30%part70%sol.csv':
        {
            'visc': 0.0134,
            'visc_error': 0.0051,
        },
    '85%glys_30%part70%sol2.csv':
        {
            'visc': 0.0134,
            'visc_error': 0.0051,
        },
    '75%glys_10%part90%sol1.csv':
        {
            'visc': 0.10163,
            'visc_error': 0.04941,
        },
    '75%glys_10%part90%sol2.csv':
        {
            'visc': 0.10163,
            'visc_error': 0.04941,
        },
    '60%glys_10%part90%sol1.csv':
        {
            'visc': 0.00942,
            'visc_error': 0.001982
            ,
        },
    '60%glys_10%part90%sol2.csv':
        {
            'visc': 0.00942,
            'visc_error': 0.001982
            ,
        },
    '40%glys_10%part90%sol1.csv':
        {
            'visc': 0.0035713,
            'visc_error': 0.0003559
            ,
        },
    '40%glys_10%part90%sol2.csv':
        {
            'visc': 0.0035713,
            'visc_error': 0.0003559
            ,
        },
    '20%glys_10%part90%sol1.csv':
        {
            'visc': 0.0016941,
            'visc_error': 0.0000638
            ,
        },
    '20%glys_10%part90%sol2.csv':
        {
            'visc': 0.0016941,
            'visc_error': 0.0000638
            ,
        },
    '100%water.csv':
        {
            'temp': 296,    #23 deg. celcius
            'temp_error': 1,
            'visc': 0.00093505,      #100% water at 23 deg celcius
            'visc_error': 0.00001
        },
    '100%water2.csv':
        {
            'temp': 296,  # 23 deg. celcius
            'temp_error': 1,
            'visc': 0.00093505,  # 100% water at 23 deg celcius
            'visc_error': 0.00001
        },
    '40deg-take2.csv':
        {
            'temp': 40+CEL_TO_KEL,
            'temp_error': 0.5,
         },
    '45deg-take2.csv':
        {
            'temp': 45+CEL_TO_KEL,
            'temp_error': 0.5,
         },
    '33 deg.csv':
        {
            'temp': 33+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '38 deg.csv':
        {
            'temp': 38+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '52.0.csv':
        {
            'temp': 52+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '27.9 deg.csv':
        {
            'temp': 27.9+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '50deg-take2.csv':
        {
            'temp': 50+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '62deg':
        {
            'temp': 62+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '67deg.csv':
        {
            'temp': 67+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '57 deg.csv':
        {
            'temp': 57+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '43 deg.csv':
        {
            'temp': 43+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '48.0.csv':
        {
            'temp': 48+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '57deg-take2.csv':
        {
            'temp': 57+CEL_TO_KEL,
            'temp_error': 0.5,
        },
    '62deg.csv':
        {
            'temp': 62+CEL_TO_KEL,
            'temp_error': 0.5,
        },


}

# ============= FUNCTIONS ============= #


def get_radius_error(radius):
    """
    takes into account z axis errors
    :param radius:
    :return:
    """
    return 0.05*radius

def get_chi_sq(table3_path):
    """
    returns reduced chi_sq
    :param table3_path: DataFrame with columns:
    :return: chi_squared
    """
    df = pd.read_csv(table3_path)

    sum = 0
    for i in range(df.particle.size):
        sum += np.square((df.iloc[i].coef_inverse - df.iloc[i].theory_val) / (df.iloc[i].theory_err+df.iloc[i].std_err))
    return sum/df.particle.size


def fill_table3_from_data_dir(data_dirname, part_select_dict, table3_path):
    """
    Takes a path to a directory with data files, and fills a table3 style data file with all the data in the dir.
    Make sure that the data file names are identical to the 'video' column in the corresponding particle selection
    table (the name of the the table doesn't matter), and that the particles in that table have either 0 or 1
    in the 'usable' column, and their correct size in the 'actual_size' column.
    If any selection data is missing, all particles in the corresponding .csv data file will be included in the
    analysis, with their trackpy size as their radius (in microns). Pay attention to warning messages throughout
    the run.
    For data files with non-default viscosity and temperature, these need to be changed in the ENVIRONMENT_VARIABLES
    dictionary.
    :param part_select_dict
    """
    # This is summarized data (video, particle, r^sq, t, radius, temp etc.) from an entire directory
    data = analyzer.get_distance_sq_data_from_dir(data_dirname, part_select_dict)
    for video in data['video'].unique():
        vid_data = data.loc[data['video'] == video]
        for particle in vid_data['particle'].unique():
            part_sum = vid_data.loc[vid_data['particle'] == particle]
            append_table3(part_sum, table3_path)


def append_table3(part_sum, table3_path, particle_size=0):
    """
    :param data: dataframe
    :param particle: int
    :param table3_path: string
    :param particle_size: int
    :return:
    """
    first_row = part_sum.iloc[0]
    radius = first_row['radius']
    radius_err = get_radius_error(radius)
    particle = first_row['particle']

    # write data to table_3
    c, s, std_err = get_regression_table2(part_sum)
    if NORMALIZE_COEF:
        std_err_inverse = get_std_err_inverse_normalize(c,std_err,radius_err,radius)
        c = c/radius
    else:
        std_err_inverse = get_function_err_inverse(c, std_err)

    theory_val, theory_err = theoretical_model.get_estimated_inverse_slope(
        part_sum.iloc[0].temp, part_sum.iloc[0].temp_error, part_sum.iloc[0].visc,
        part_sum.iloc[0].visc_error, radius, radius_err)

    if IS_WEEK3:
        theory_val, theory_err = theoretical_model.get_estimated_slope(
            part_sum.iloc[0].temp, part_sum.iloc[0].temp_error, part_sum.iloc[0].visc,
            part_sum.iloc[0].visc_error, radius, radius_err)

        df = pd.DataFrame([[particle, c, s, radius, radius_err, std_err, theory_err, theory_val,
                            first_row.visc, first_row.visc_error, first_row.temp, first_row.temp_error,
                            first_row.video]],
                          columns=['particle', 'coef_inverse', 'score', 'radius', 'radius_err', 'std_err',
                                   'theory_err', 'theory_val',
                                   'visc', 'visc_error', 'temp', 'temp_error', 'video'])
    else:
        df = pd.DataFrame([[particle, 1/c, s, radius,radius_err, std_err_inverse, theory_err, theory_val,
                        first_row.visc, first_row.visc_error,first_row.temp,first_row.temp_error,
                        first_row.video]],
                      columns=['particle', 'coef_inverse', 'score', 'radius','radius_err', 'std_err',
                               'theory_err', 'theory_val',
                               'visc','visc_error','temp','temp_error','video'])
    # sum_file = '100%water.table3.csv'
    if os.path.isfile(table3_path):
        with open(table3_path, 'a') as f:
            df.to_csv(f, header=None)
    else:
        with open(table3_path, 'a') as f:
            df.to_csv(f)


def get_std_err_normalize(coef,std_err,radius_err,radius):
    return((radius*std_err)**2*(radius_err*coef)**2)**0.5

def get_std_err_inverse_normalize(coef,std_err,radius_err,radius):
    return ((radius*coef)**-2)*(std_err**2+radius_err**2)**0.5


def get_function_err_inverse(coef, std_err):
    return (coef**-2)*std_err


def show_annotated_first_frame(data, img_dir_path):
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


def get_regression_table3_week2(table3_path):
    """
    per particle:
    creates linear regression for r_sq vs t,
    plots graph
    returns coeff and r^2 score and stderr
    :param part_sum: dataframe with r_sq, time_gap columns
    :return: coeff: array size 1, score: int
    """
    df = pd.read_csv(table3_path)
    mod = smf.ols(formula='coef_inverse ~ visc - 1', data=df)
    res = mod.fit()
    return res.params[0], res.rsquared, res.bse[0]
    # return regr.coef_[0], regr.score(x, y), stderr

def get_regression_table3_week3(table3_path):
    """
    per particle:
    creates linear regression for r_sq vs t,
    plots graph
    returns coeff and r^2 score and stderr
    :param part_sum: dataframe with r_sq, time_gap columns
    :return: coeff: array size 1, score: int
    """
    df = pd.read_csv(table3_path)
    mod = smf.ols(formula='coef_inverse ~ temp - 1', data=df)
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
    part_sum = analyzer.get_particle_sq_distance_data(data[data.particle == particle])
    # write data to table_2
    part_sum['particle'] = particle
    with open(table2_path, 'a') as f:
        part_sum.to_csv(f, header=None)


def add_environment_variables(data, filepath):
    """
    Fills a data file with temperature and viscosity data, according to a dict at the top of this file.
    The name of the data file (for example, temp30.csv) must be given. If it is not found, or if any
    variable is missing - inserts default values.
    """
    file_dict = dict()
    filename = os.path.basename(filepath)
    # assert(filename in ENVIRONMENT_VARIABLES.keys()), 'error - can"t find {} in dict '.format(filename)
    if filename in ENVIRONMENT_VARIABLES.keys():
        file_dict = ENVIRONMENT_VARIABLES[filename]
    for varname in DEFAULT_ENV_VARIABLES.keys():
        if varname in file_dict.keys():
            var_value = file_dict[varname]
        else:
            print("Warning: using default " + str(varname) + " for " + str(filename))
            var_value = DEFAULT_ENV_VARIABLES[varname]
        data[varname] = var_value
    return data


def get_data(raw_data_path, part_select_dict):
    """
    :param raw_data_path: string
    :return:
    """
    print("Processing file " + raw_data_path)
    data_filename = os.path.splitext(os.path.basename(raw_data_path))[0]
    data = pd.read_csv(raw_data_path)
    data = tp.link_df(data, analyzer.MAX_PIXELS_BW_FRAMES, memory=analyzer.TRACKING_MEMORY)
    print(str(len(data.particle.unique())) + " initial trajectories")
    data = tp.filter_stubs(data, analyzer.MIN_TRACK_LENGTH)
    print(str(len(data.particle.unique())) + " trajectories span at least " + str(analyzer.MIN_TRACK_LENGTH) + " frames" )
    data = analyzer.filter_particles_and_add_actual_size(data, data_filename, part_select_dict)
    print(str(len(data.particle.unique())) + " selected particles left")
    drift = tp.compute_drift(data)
    data = tp.subtract_drift(data, drift)
    data = analyzer.cancel_avg_velocity_drift(data)
    data = add_environment_variables(data, raw_data_path)
    return data


def plot_table_3(table3_path):
    df = pd.read_csv(table3_path)
    plt.errorbar(df.radius, df.coef_inverse, df.std_err, xerr=df.radius_err, fmt="o", capsize=4)
    # plt.xlabel('radius in micron')
    # plt.ylabel('coef_inverse = (3*pi*eta*r)/(2*k*T) in micron squared per sec')

    plt.errorbar(df.radius,df.theory_val/df.radius,df.theory_err, xerr=df.radius_err, fmt="o", capsize=4)
    plt.ylabel('theory_val = (3*pi*eta*r)/(2*k*T) in micron squared per sec')
    plt.show()


def get_n_sigma_week2(table3_path):
    df = pd.read_csv(table3_path)
    first_row = df.iloc[0]
    fit_slope, s, fit_err = get_regression_table3_week2(table3_path)
    theory_slope = first_row.theory_val / first_row.visc
    # theory_err = ((first_row.theory_err/first_row.visc)**2*(theory_slope*(first_row.visc_error/10000)/(first_row.visc**2))**2)**0.5
    theory_err = first_row.theory_err
    return abs(fit_slope-theory_slope)/(theory_err+fit_err)

def get_n_sigma_week3(table3_path):
    df = pd.read_csv(table3_path)
    first_row = df.iloc[0]
    fit_slope, s, fit_err = get_regression_table3_week3(table3_path)
    theory_slope = first_row.theory_val / first_row.temp
    theory_err = ((first_row.theory_err/first_row.temp)**2*(theory_slope*(first_row.temp_error)/(first_row.temp**2))**2)**0.5
    # theory_err = first_row.theory_err
    return abs(fit_slope-theory_slope)/(theory_err+fit_err)

NORMALIZE_COEF = False
IS_WEEK3 = False
RAW_DATA_PATH = 'data/3.csv'
TABLE2_PATH = '100%water.table2.csv'
TABLE3_PATH = 'table3_week1.csv'
SELECTION_DIRPATH = './selected_particles'

if __name__ == '__main__':

    # This can be replaced by a call to 'fill_table3..'
    # sel_dict = get_selected_particles_dict(SELECTION_DIRPATH)
    # particles = sel_dict['3.0']
    # if len(particles)>0:
    #     data = get_data(RAW_DATA_PATH)
    #     for p in particles:
    #         if p in data.particle.unique():
    #             append_table3(data, p, TABLE3_PATH)
    #             print(get_chi_sq(TABLE3_PATH))

    # d = analyzer.get_selected_particles_dict('./selected_particles')
    # fill_table3_from_data_dir('data',d,TABLE3_PATH)

    plot_table_3(TABLE3_PATH)
    # # print(get_chi_sq(TABLE3_PATH))
    # print("c, s, std_err{}".format(get_regression_table3_week3(TABLE3_PATH)))
    # print(get_n_sigma_week3(TABLE3_PATH))