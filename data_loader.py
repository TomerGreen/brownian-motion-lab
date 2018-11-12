import matplotlib as mpl
import matplotlib.pyplot as plt
import trackpy as tp
import cv2, os, zipfile, pims

FRAME_NAME = 'frame'
PARTICLE_SIZE = 71
MIN_MASS = 100
PERCENTILE = 99.2
VIDEO_DIRNAME = 'videos'
RAW_DATA_DIRNAME = 'data'


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


def save_data_from_dir(dirpath):
    for root, dirs, files in os.walk(dirpath):
        for filename in files:
            if filename.endswith('.avi'):
                out_dir = extract_frames(root + '/' + filename)
                save_data(out_dir, RAW_DATA_DIRNAME)

if __name__ == '__main__':
    save_data_from_dir(VIDEO_DIRNAME)