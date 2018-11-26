import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
import mmain
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import functools
import theoretical_model as tm


MAX_TIME_GAP = 50
MIN_TRACK_LENGTH = 50
MAX_PIXELS_BW_FRAMES = 5
TRACKING_MEMORY = 3


def cancel_avg_velocity_drift(data):
    frame_count_lambda = lambda x: x - x.min()
    data['frame_count'] = data.groupby('particle')['frame'].transform(frame_count_lambda)
    data['last_frame_count'] = data.groupby('particle')['frame_count'].transform(max)
    last_frame_coors = data[data['frame_count'] == data['last_frame_count']][['particle', 'x', 'y', 'last_frame_count']]
    last_frame_coors = last_frame_coors.rename({'x': 'last_x_coor', 'y': 'last_y_coor'}, axis='columns')
    first_frame_coors = data[data['frame_count'] == 0][['particle', 'x', 'y']]
    first_frame_coors = first_frame_coors.rename({'x': 'first_x_coor', 'y': 'first_y_coor'}, axis='columns')
    vel_table = pd.merge(first_frame_coors, last_frame_coors, on='particle')
    vel_table['avg_x_vel'] = (vel_table['last_x_coor'] - vel_table['first_x_coor']) / vel_table['last_frame_count']
    vel_table['avg_y_vel'] = (vel_table['last_y_coor'] - vel_table['first_y_coor']) / vel_table['last_frame_count']
    data = pd.merge(data, vel_table[['particle', 'avg_x_vel', 'avg_y_vel']], on='particle', how='left')
    data['x'] = data['x'] - (data['avg_x_vel'] * data['frame_count'])
    data['y'] = data['y'] - (data['avg_y_vel'] * data['frame_count'])
    return data


# REDUNDANT
"""
def get_linked_data_without_drift(datafile):
    data = pd.read_csv(datafile)
    data = tp.link_df(data, MAX_PIXELS_BW_FRAMES, memory=TRACKING_MEMORY)
    data = tp.filter_stubs(data, MIN_TRACK_LENGTH)
    print('Found ' + str(data['particle'].nunique()) + ' particles')
    data = cancel_avg_velocity_drift(data)
    return data
    """


def get_distance_sq_data(datafile):
    """
    Takes a tracking data file and returns summarized data per particle.
    :param datafile: a raw tracking data file, without linking or drift-cancelling.
    :return: a data frame with columns: particle, r_sq, time_gap and residual.
    """
    data = mmain.get_data(datafile)
    print(data.head())
    r_sq_data = pd.DataFrame()
    for particle in data.particle.unique():
        part_data = data[data.particle == particle]
        part_sum = get_particle_sq_distance_data(part_data)
        part_sum['particle'] = particle
        part_sum = add_residuals_to_particle_summary(part_sum)
        r_sq_data = r_sq_data.append(part_sum)
    return r_sq_data


def get_mean_residual_by_time_frame(data, cut_quantile):
    """
    Takes sumarrized data per particle and returns the residual averaged by particle, per time frame.
    This will enable us to examine systematic shift from a linear fit, by time.
    :param data: summarized data per particle with r_sq and residual per time frame
    :param cut_quantile: a quantile like 0.05 or 0.1. Residuals below that quantile or above 1-cut_quantile
    will be ignored.
    :return: mean of residuals per time frame.
    """
    top_quant_col_name = 'quantile_' + str(1-cut_quantile)
    bottom_quant_col_name = 'quantile_' + str(cut_quantile)
    data[bottom_quant_col_name] = data.groupby('time_gap')['residual'].transform(lambda x: x.quantile(cut_quantile))
    data[top_quant_col_name] = data.groupby('time_gap')['residual'].transform(lambda x: x.quantile(1-cut_quantile))
    data = data[(data['residual'] <= data[top_quant_col_name]) & (data['residual'] >= data[bottom_quant_col_name])]
    res_by_time = data.groupby('time_gap').agg('mean')[['residual']]
    return res_by_time


def get_particle_sq_distance_data(part_data):
    """
    Gets rows of data relevant to one particle and returns summarized data IN PHYSICAL QUANTITIES for that particle
    :param part_data: A row that includes (at least) particle number, size, x and y coordinates and frame number.
    :return: A data frame with particle number, particle radius, and <r^2> in m^2 per time gap in seconds.
    """

    def _get_sq_distance(row):
        return row['x'] ** 2 + row['y'] ** 2

    part_num = int(part_data.iloc[0]['particle'])
    part_rad = part_data['size'].mean() * tm.PIXEL_LENGTH_IN_MICRONS
    rad_error = np.sqrt(tm.RELATIVE_PIXEL_NUM_ERROR ** 2 + (tm.PIXEL_LENGTH_ERROR / tm.PIXEL_LENGTH_IN_MICRONS) ** 2) * part_rad
    result = pd.DataFrame(columns=['time_gap', 'r_sq'])
    for gap in range(1, MAX_TIME_GAP+1):
        gap_data = part_data.iloc[::gap, :]    # the x,y position of every nth row.
        diff_data = gap_data.diff()
        # Each row shows the squared difference from the previous one.
        distance_sq_series = diff_data.apply(_get_sq_distance, axis='columns')
        mean_distance_sq = distance_sq_series.mean()
        result = result.append({'time_gap': gap, 'r_sq': mean_distance_sq}, ignore_index=True)
        result = result.dropna()
    result['time_gap'] = result['time_gap'] * tm.SECONDS_PER_FRAME
    result['r_sq'] = result['r_sq'] * (tm.PIXEL_LENGTH_IN_MICRONS ** 2)
    result['particle'] = part_num
    result['radius'] = part_rad     # These are already in meters.
    result['radius_error'] = rad_error
    # Copies the temp and viscosity data from the argument data.
    for varname in mmain.DEFAULT_ENV_VARIABLES.keys():
        result[varname] = part_data.iloc[0][varname]
    return result


def add_residuals_to_particle_summary(part_sum):
    """
    Gets distance squared vs. time frame data for a particle, and returns a data frame of residuals
    per time frame.
    """

    # ===== FOR LINEAR FIT USE THIS ===== #

    lin_coeff = mmain.get_regression_table2(part_sum)[0]
    part_sum['residual'] = part_sum['r_sq'] - part_sum['time_gap'] * lin_coeff


    # ===== FOR LINEAR AND EXPONENTIAL FIT USE THIS ===== #

    #params = get_linear_plus_exp_fit(part_sum)
    #print(params)
    #part_sum['residual'] = part_sum['r_sq'] - part_sum['time_gap'].apply(lin_plus_exp_fit_function,
    #                                                                    args=(params[0], params[1], params[2]))

    return part_sum


def linear_fit_function(t, coeff):
    return t * coeff


def lin_plus_exp_fit_function(t, m, c, a):
    #print("Applying parameters: " + str(t) + ", " + str(m) + ", " + str(c) + ", " + str(a))
    #print("just exponent is: " + str(np.exp(-a*t)))
    expression = m * t + (c / a) * (1 - np.exp(-a * t))
    return expression


def get_linear_plus_exp_fit(part_sum):
    params, covariance = opt.curve_fit(lin_plus_exp_fit_function, part_sum['time_gap'], part_sum['r_sq'], maxfev=1500)
    return params


if __name__ == '__main__':
    r_sq_data = get_distance_sq_data('data/8.csv')
    #res_data = get_mean_residual_by_time_frame(r_sq_data, 0.1)
    #plt.scatter(res_data.index.values, res_data['residual'])
    #plt.show()