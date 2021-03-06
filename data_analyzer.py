import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
import mmain
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import functools
import theoretical_model as tm
import os


MIN_TIME_GAP = 0.4
MAX_TIME_GAP = 15
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


def get_selected_particles_dict(selection_dirpath):
    """
    Takes a path to a particle selection data directory, and returns a particle selection dictionary.
    :param selection_dirpath: a path to a directory containing .xlsx or .csv files with particle selection
    data, meaning that every row represents a particle, and contains the video name, particle numbers and a 'usable'
    column with 1's and 0's
    :return: A dictionary with video names and lists of particle details dictionaries:
    {
        data_file1:
        [
            {
                name: 1
                initial_coordinates: (100, 200)
                actual_size: 13.5
            },
            {
                name:2
                ...
            }
        ],
        data_file2:
        [
        ...
        ]
    }
    """
    selection_dict = dict()
    for root, dirs, files in os.walk(selection_dirpath):
        for filename in files:
            if filename.endswith('.csv'):
                sel_file_data = pd.read_csv(root + '/' + filename)
            elif filename.endswith('.xlsx'):
                sel_file_data = pd.read_excel(root + '/' + filename)
            else:
                continue
            vidname = str(sel_file_data.iloc[0]['video'])
            actual_size_exists = 'actual_size' in sel_file_data.columns
            if not actual_size_exists:
                print("Warning: no actual_size column in " + vidname + " selection table. Using trackpy sizes.")
            usable_parts_data = sel_file_data.loc[sel_file_data['usable'] == 1]
            part_details_list = list()
            for index, row in usable_parts_data.iterrows():
                part_details = dict()
                part_details['name'] = row['particle']
                part_details['initial_coordinates'] = (row['x'], row['y'])      # A tupple of x and y coors in frame 0.
                if actual_size_exists:
                    part_details['actual_size'] = float(row['actual_size'])
                else:
                    part_details['actual_size'] = row['size']
                part_details_list.append(part_details)
            selection_dict[vidname] = part_details_list
    return selection_dict


def filter_particles_and_add_actual_size(data, data_filename, select_part_dict):
    """
    Takes data from one file, the filename, and a selection dictionary, and returns the data for
    selected particles only, and with actual_size. For files that are not in the selection dictionary,
    the data is unchanged. If there is no actual_size column, the dictionary already takes the trackpy size.
    If a particle does not appear on frame 0, it is filtered out.
    """
    result = pd.DataFrame()
    data['actual_size'] = data['size']
    data_filename = os.path.basename(data_filename)
    # Handles numerical names that were butchered by format transformation.
    # try:
    #     data_filename = str(float(data_filename))
    # except ValueError:
    #     pass
    # Ignores data files that are not in the selection dict.
    assert (data_filename in select_part_dict.keys()), \
        'Error {}  data_filename is not in the selection dictionary.'.format(data_filename)
    if data_filename not in select_part_dict.keys():
        result = data
        print("Warning: file " + data_filename
            + " is not in the selection dictionary. all particles in it will be included, with original sizes.")
    else:
        for particle in data.particle.unique():
            part_data = data[data['particle'] == particle].copy()
            part_frame_zero_row = part_data.loc[part_data['frame'] == 0]
            # Skips particles that are not in the first frame.
            if not part_frame_zero_row.empty:
                part_frame_zero_row = part_frame_zero_row.iloc[0]
                # The x and y coors of the current particle in frame 0.
                part_x = part_frame_zero_row['x']
                part_y = part_frame_zero_row['y']
                for part_details_dict in select_part_dict[data_filename]:
                    x = part_details_dict['initial_coordinates'][0]
                    y = part_details_dict['initial_coordinates'][1]
                    actual_size = part_details_dict['actual_size']
                    if x - 1 <= part_x <= x + 1 and y - 1 <= part_y <= y + 1:
                        part_data['actual_size'] = actual_size
                        result = result.append(part_data)
                        continue
    return result


def get_distance_sq_data_from_dir(data_dirpath, select_part_dict):
    """
    Returns summarized data for an entire folder. Actually appends data frames like you get from
    get_distance_sq_data_from_file, and adds a column with the filename called 'video'.
    """
    result = pd.DataFrame()
    for root, dirs, files in os.walk(data_dirpath):
        for filename in files:
            if filename.endswith('.csv'):
                data_from_file = get_distance_sq_data_from_file(root + '/' + filename, select_part_dict)
                data_filename = os.path.splitext(filename)[0]
                data_from_file['video'] = data_filename
                result = result.append(data_from_file)
    return result


def get_distance_sq_data_from_file(datafile, select_part_dict):
    """
    Takes a tracking data file and returns summarized data per particle.
    :param datafile: a raw tracking data file, without linking or drift-cancelling.
    :param select_part_dict: a particle selection dictionary created by the function above.
    :return: a data frame with columns: particle, size r_sq, time_gap and residual, along with experiment
    variables that are passed from the get_data function.
    """
    data = mmain.get_data(datafile, select_part_dict)
    r_sq_data = pd.DataFrame()
    for particle in data['particle'].unique():
        part_data = data[data['particle'] == particle]
        part_sum = get_particle_sq_distance_data(part_data)
        part_sum['particle'] = particle
        part_sum = add_residuals_to_particle_summary(part_sum)
        r_sq_data = r_sq_data.append(part_sum)
    return r_sq_data


def get_fit_r_sq_hist(data):
    fit_r_sq_per_part = data.groupby(['video', 'particle'], as_index=False).agg(np.mean)
    print(fit_r_sq_per_part.head())
    fit_r_sq_per_part = fit_r_sq_per_part[fit_r_sq_per_part['fit_r_sq'] > 0.85]['fit_r_sq']
    plt.hist(fit_r_sq_per_part, width=0.003, bins=40)
    print("Mean r^2 value for fitting " + str(fit_r_sq_per_part.shape[0]) + " particles: " + str(np.mean(fit_r_sq_per_part)))
    plt.suptitle("R^2 Value Histogram After Fitting All Particles")
    plt.xlabel("Linear Fit R^2 Value")
    plt.show()


def get_mean_residual_by_time_frame(data, cut_quantile):
    """
    Takes summarized data per particle and returns the residual averaged by particle, per time frame.
    This will enable us to examine systematic shift from a linear fit, by time.
    :param data: summarized data per particle with r_sq and residual per time frame
    :param cut_quantile: a quantile like 0.05 or 0.1. Residuals below that quantile or above 1-cut_quantile
    will be ignored.
    :return: mean of residuals per time frame.
    """
    top_quant_col_name = 'quantile_' + str(1-cut_quantile)
    bottom_quant_col_name = 'quantile_' + str(cut_quantile)
    print(data.head())
    data[bottom_quant_col_name] = data.groupby('time_gap')['relative_residual'].transform(lambda x: x.quantile(cut_quantile))
    data[top_quant_col_name] = data.groupby('time_gap')['relative_residual'].transform(lambda x: x.quantile(1-cut_quantile))
    print(data.shape)
    data = data[(data['relative_residual'] <= data[top_quant_col_name]) & (data['relative_residual'] >= data[bottom_quant_col_name])]
    print(data.shape)
    res_by_time = data.groupby('time_gap', as_index=False).agg(np.mean)
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
    part_rad = part_data['actual_size'].mean() * tm.MICRONS_PER_PIXEL
    rad_error = np.sqrt(tm.RELATIVE_PIXEL_NUM_ERROR ** 2 + (tm.PIXEL_LENGTH_ERROR / tm.MICRONS_PER_PIXEL) ** 2) * part_rad
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
    result['r_sq'] = result['r_sq'] * (tm.MICRONS_PER_PIXEL ** 2)
    result['particle'] = part_num
    result['radius'] = part_rad     # These are already in microns.
    result['radius_error'] = rad_error
    # Copies the temp and viscosity data from the argument data.
    for varname in mmain.DEFAULT_ENV_VARIABLES.keys():
        result[varname] = part_data.iloc[0][varname]
    # plt.figure().suptitle('Sample <r^2> by time for one particle without drift')
    # plt.xlabel('Time (s)')
    # plt.ylabel('<r^2> (10^-12 m^2)')
    # plt.scatter(result['time_gap'], result['r_sq'])
    # plt.show()
    return result


def add_residuals_to_particle_summary(part_sum):
    """
    Gets distance squared vs. time frame data for a particle, and returns a data frame of residuals
    per time frame.
    """

    # ===== FOR LINEAR FIT USE THIS ===== #

    lin_fit_res = mmain.get_regression_table2(part_sum)
    lin_coeff = lin_fit_res[0]
    rsquared = lin_fit_res[1]
    part_sum['residual'] = part_sum['r_sq'] - part_sum['time_gap'] * lin_coeff
    part_sum['relative_residual'] = part_sum['residual']/part_sum['r_sq']
    part_sum['fit_r_sq'] = rsquared

    # plt.bar(part_sum['time_gap'], part_sum['residual'], width=0.8)  # (tm.PIXEL_LENGTH_IN_MICRONS**2)
    # plt.suptitle("Particle residual from linear fit by time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("<r^2> (10^-12 m^2)")
    # plt.show()


    # ===== FOR NON LINEAR FIT USE THIS ===== #

    # params = get_lin_fit(part_sum)
    # print(params)
    # part_sum['residual'] = part_sum['r_sq'] - part_sum['time_gap'].apply(lin_fit_function,
    #                                                                     args=(params[0],))
    # part_sum['relative_residual'] = part_sum['residual'] / part_sum['r_sq']

    return part_sum


def lin_fit_function(t, a):
    return  a * t

def get_lin_fit(part_sum):
    params, covariance = opt.curve_fit(square_fit_function, part_sum['time_gap'], part_sum['r_sq'], maxfev=1500)
    return params

def square_fit_function(t, a, b):
    return a * t + b * (t**2)


def get_square_fit(part_sum):
    params, covariance = opt.curve_fit(square_fit_function, part_sum['time_gap'], part_sum['r_sq'], maxfev=1500)
    return params


def quad_fit_function(t, a, b, c, d):
    return a * t + b * (t**2) + c * (t**3) + d * (t**4)


def get_quad_fit(part_sum):
    params, covariance = opt.curve_fit(quad_fit_function, part_sum['time_gap'], part_sum['r_sq'], maxfev=1500)
    return params


if __name__ == '__main__':
    sel_dict = get_selected_particles_dict('./selected_particles')
    data = get_distance_sq_data_from_dir('data', sel_dict)
    get_fit_r_sq_hist(data)
    #r_sq_data = get_distance_sq_data_from_dir('data', sel_dict)
    # res_data = get_mean_residual_by_time_frame(data, 0.05)
    # plt.bar(res_data['time_gap'], res_data['relative_residual'], width=0.08)
    # plt.suptitle("Average relative residual by time - ignoring small time gaps")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Relative residual (no units)")
    # plt.show()
