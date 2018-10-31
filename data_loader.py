import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import os
import zipfile


def zipdir(path, zipname):
    """Creates a zip out of a directory. ziph is a zipfile handle."""
    zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()


def frame_capture(path):
    """Extracts frames from video."""
    dirname, filename = os.path.split(path)
    filename_no_ext = os.path.splitext(filename)[0]     # The filename without extension.
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = True
    while success:
        success, image = vidObj.read()
        frame_filename = dirname + '/' + filename_no_ext + '/' + "frame%d.jpg" % count
        # Saves the frames with frame-count
        print('writing to ' + frame_filename)
        cv2.imwrite(frame_filename, image)
        count += 1

if __name__ == '__main__':
    frame_capture('videos/1.avi')