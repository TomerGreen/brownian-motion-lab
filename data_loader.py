import matplotlib as mpl
import matplotlib.pyplot as plt
import trackpy as tp
import pandas as pd
import cv2, os, zipfile, pims


# ============= Tracking Parameters =============#

FRAME_NAME = 'frame'
PARTICLE_SIZE = 71      # Updated 5.11
MIN_MASS = 100          # Updated 5.11
PERCENTILE = 99.2       # Updated 5.11
VIDEO_DIRNAME = 'week_3_videos'
RAW_DATA_DIRNAME = 'week_3_data'


# ============= Linking Parameters =============#

MAX_PIXELS_BW_FRAMES = 5
TRACKING_MEMORY = 3


# ============= Function Library =============#

def zipdir(path, zipname):
    """Creates a zip out of a directory. ziph is a zipfile handle."""
    zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()


def extract_frames(path):
    """
    Extracts frames from video and saves them in the same location inside a directory
    with the same name as the video.
    :return: the dirpath to the frame directory.
    """
    dirname, filename = os.path.split(path)
    filename_no_ext = os.path.splitext(filename)[0]     # The filename without extension.
    out_dirname = dirname + '/' + filename_no_ext
    try:
        os.mkdir(out_dirname)
    except OSError:     # In case a directory with the video's name already exists.
        pass
    vidObj = cv2.VideoCapture(path)
    count = 1
    success = True
    while success:
        success, image = vidObj.read()
        frame_filename = out_dirname + '/' + FRAME_NAME + "%d.jpg" % count
        # Saves the frames with frame-count
        if (count == 1 or count%100 == 0):
            print('Writing frame ' + str(count) + ' to ' + out_dirname)
        cv2.imwrite(frame_filename, image)
        count += 1
    print('Finished writing ' + str(count) + ' frames to ' + out_dirname)
    return out_dirname


def save_data(frame_dir, save_to_dir):
    """
    Takes a dirpath containing video frames and extracts the data.
    :param frame_dir: the directory containing the frames.
    :param save_to_dir: the directory path where we save our data, without the last '/'.
    :return:
    """
    print("Getting frame data from " + frame_dir)
    frames = pims.ImageSequence(frame_dir + '/' + FRAME_NAME + '*.jpg', as_grey=True)
    try:
        data = tp.batch(frames[:-2], PARTICLE_SIZE, invert=True, minmass=MIN_MASS, percentile=PERCENTILE)
    except OSError:
        pass
    frame_dirname = os.path.basename(frame_dir)
    out_filepath = save_to_dir + '/' + frame_dirname + '.csv'
    print("Writing frame data to " + out_filepath)
    data.to_csv(out_filepath)
    #plt.figure()
    #tp.annotate(data, frames[0])


def save_data_from_dir(vid_dirpath, data_dirpath):
    """
    Takes the paths of a directory containing videos and a directory into which to save the data.
    Extracts the tracking data from every .avi file in the video dir to a CSV with the same name
    in the data dir. Overwrites existing CSV's if they have the same name.
    :param vid_dirpath: the video directory path.
    :param data_dirpath: the data directory path.
    :return:
    """
    for root, dirs, files in os.walk(vid_dirpath):
        for filename in files:
            if filename.endswith('.avi'):
                out_dir = extract_frames(root + '/' + filename)
                save_data(out_dir, data_dirpath)

def add_linking_data_to_csv(datafile):
    """
    Adds a 'particle' column to tracking data CSV according to global parameters.
    Actually replaces the existing CSV with another one which includes particle linking.
    Does nothing if datafile does not have a .csv extension.
    :param datafile: A path to the file to be processed.
    """
    if datafile.endswith('.csv'):
        data = pd.read_csv(datafile)
        data = tp.link_df(data, MAX_PIXELS_BW_FRAMES, memory=TRACKING_MEMORY)
        data.to_csv(datafile)


def add_linking_data_to_dir(data_dirpath):
    """
    Same as add_linking_data_to_csv but for an entire directory containing CSV's.
    :param data_dirpath: the CSV directory
    """
    for root, dirs, files in os.walk(data_dirpath):
        for file in files:
            datafile = root + '/' + file
            print("Linking " + str(datafile))
            add_linking_data_to_csv(datafile)   # Takes care of validating .csv extension.


if __name__ == '__main__':
    save_data_from_dir(VIDEO_DIRNAME, RAW_DATA_DIRNAME)
    #add_linking_data_to_dir(RAW_DATA_DIRNAME)