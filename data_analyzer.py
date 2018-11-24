import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


MAX_TIME_GAP = 50
MIN_TRACK_LENGTH = 50


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
    print(data.head(100))

    # Older attempt
    """
    # Adds two empty columns for x and y average velocities.
    data['avg_x_vel'] = np.nan
    data['avg_y_vel'] = np.nan
    for particle in data.particle.unique():
        part_data = data[data.particle == particle]
        first_frame = part_data['frame'].min()
        last_frame = part_data['frame'].max()
        first_frame_row = part_data[part_data['frame'] == first_frame]
        first_x_coor = first_frame_row.at(0, 'x')
        first_y_coor = first_frame_row.at(0, 'y')
        last_frame_row = part_data[part_data['frame'] == last_frame]
        last_x_coor = last_frame_row.at(0, 'x')
        last_y_coor = last_frame_row.at(0, 'y')
        # The average x and y velocities in distance per frame.
        x_drift_per_frame = (last_x_coor-first_x_coor)/(last_frame-first_frame)
        y_drift_per_frame = (last_y_coor-first_y_coor)/(last_frame-first_frame)
        data.loc[data.particle == particle, 'avg_x_vel'] = x_drift_per_frame
        data.loc[data.particle == particle, 'avg_y_vel'] = y_drift_per_frame
        data.loc[data.particle == particle, 'frames_from_start'] = data.loc[data.particle == particle, 'frame'] - first_frame
    data['x'] = data['x'] - (data['frames_from_start'] * data['avg_x_vel'])
    data['y'] = data['y'] - (data['frames_from_start'] * data['avg_y_vel'])
    data = data.sort(['particle', 'frame'], ascending=[True, True])
    print(data.head(500))
    return data
    """


def get_distance_sq_data(datafile):
    data = pd.read_csv(datafile)

    data = tp.filter_stubs(data, MIN_TRACK_LENGTH)
    data = cancel_avg_velocity_drift(data)
    print('Found ' + str(data['particle'].nunique()) + ' particles')
    data = cancel_avg_velocity_drift(data)
    for particle in data.particle.unique():
        part_data = data[data.particle == particle].loc[:, ['x', 'y']]    # x,y data for current particle.
        part_sum = get_particle_sq_distance_data(part_data)
        plt.plot(part_sum['frame_gap'], part_sum['r_sq'])
    plt.show()


def get_particle_sq_distance_data(part_data):

    def _get_sq_distance(row):
        return row['x']**2 + row['y']**2

    result = pd.DataFrame(columns=['frame_gap', 'r_sq'])
    for gap in range(1, MAX_TIME_GAP+1):
        gap_data = part_data.iloc[::gap, :]    # the x,y position of every nth row.
        diff_data = gap_data.diff()
        # Each row shows the squared difference from the previous one.
        distance_sq_series = diff_data.apply(_get_sq_distance, axis='columns')
        mean_distance_sq = distance_sq_series.mean()
        result = result.append({'frame_gap':gap, 'r_sq':mean_distance_sq}, ignore_index=True)
    return result



if __name__ == '__main__':
    data = pd.read_csv('0')
    cancel_avg_velocity_drift(data)