import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

MAX_PIXELS_BW_FRAMES = 5
TRACKING_MEMORY = 3
MAX_TIME_GAP = 50
MIN_TRACK_LENGTH = 50


def get_distance_sq_data(datafile):
    """
    plots time (frames) vs mean distance squared (pixels squared) for given csv file,
    multiple particles.
    :param datafile: string
    :return: void
    """
    data = pd.read_csv(datafile)
    data = tp.link_df(data, MAX_PIXELS_BW_FRAMES, memory=TRACKING_MEMORY)
    data = tp.filter_stubs(data, MIN_TRACK_LENGTH)
    print('Found ' + str(data['particle'].nunique()) + ' particles')
    for particle in data.particle.unique():
        part_data = data[data.particle == particle].loc[:,['x','y']]    # x,y data for current particle.
        part_sum = get_particle_sq_distance_data(part_data)
        plt.plot(part_sum['frame_gap'], part_sum['r_sq'])
    plt.show()


def get_particle_sq_distance_data(part_data):
    """
    returns DataFrame containing 'frame_gap' and 'r_sq', mean distance square of single particle
    :param part_data: DataFrame that contains 'x', 'y' cols, of single particle
    :return: DataFrame
    """
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
    get_distance_sq_data('D:\GDrive\Lab II\Raw Data\week 2/100%water.csv')
