import data_analyzer as analyzer
import mmain as main
import matplotlib.pyplot as plt
import matplotlib.image as img
import trackpy as tp
from matplotlib.patches import Circle
import os
from data_analyzer import *

PARTICLE_SELECTION_DATA_DIRNAME = 'particle_selection_data'


def select_particles(data, vid_name, frame_zero_path, sel_data_dirname):
    """
    Takes data from a video and a first frame filepath and then:
    1. Creates a data file with particle summary details, so you can mark on it which particles you like. The file
    is saved in a given directory name (creates one if it doesn't exist).
    2. Shows the first frame and circles one particle at a time, so you can mark it in the data file.
    Please note that running the function again will delete the previous data file, so save it in the drive.
    :param data: data from an entire video
    :param vid_name: the name of the video (for the output data file name).
    :param frame_zero_path: a path to a jpg file where the particles will be circled.
    :param sel_data_dirname: a directory name in which to save the particle selection data.
    """

    part_sel_data = data[data['frame'] == 0][['particle', 'x', 'y', 'size']]
    part_sel_data['video'] = vid_name
    # Makes a selection data directory, if one doesn't exist.
    try:
        os.mkdir(sel_data_dirname)
    except OSError:  # In case the directory exists
        pass
    sel_data_filepath = sel_data_dirname + '/' + vid_name + "_particle_selection.csv"
    print("Writing particle selection data to " + sel_data_filepath)
    part_sel_data.to_csv(sel_data_filepath)

    frame_zero = img.imread(frame_zero_path)
    for index, row in part_sel_data.iterrows():
        fig, ax = plt.subplots(1)
        circ = Circle((row['x'], row['y']), 50, fill=False, color='white')
        ax.imshow(frame_zero)
        ax.add_patch(circ)
        title = 'Particle #' + str(int(row['particle']))
        ax.set_title(title)
        plt.show()

if __name__ == '__main__':
    VIDNAME = '95%glys_10%part90%sol2'
    data = main.get_data('data/' + VIDNAME + '.csv')
    select_particles(data, VIDNAME, 'videos/' + VIDNAME + '/frame1.jpg', PARTICLE_SELECTION_DATA_DIRNAME)
