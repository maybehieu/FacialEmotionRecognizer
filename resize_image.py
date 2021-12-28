import os
import cv2
import numpy as np
import time
# path of 'datasets' folder
DATA_ROOT = ""
TRAIN_DIR = os.path.join(DATA_ROOT, "FER2013Train")
VAL_DIR = os.path.join(DATA_ROOT, "FER2013Valid")
TEST_DIR = os.path.join(DATA_ROOT, "FER2013Test")

'''
WARNING, running this code affects DIRECTLY on the ORIGINAL data, 
run at your own risks.
'''


def resize_image(dataset_dir, size=(128, 128)):
    print("Resizing images with size:({})...".format(size))
    idx = 0
    for file in os.listdir(dataset_dir):
        if file.endswith(".png"):
            idx += 1
            file_dir = os.path.join(dataset_dir, file)
            src = cv2.imread(file_dir)
            img2 = cv2.resize(src, size, interpolation=cv2.INTER_LINEAR)
            print("Resized {} complete, saving file {}...".format(file, idx))
            cv2.imwrite(file_dir, img2)
    print("Resize {} images completed. Program will shutdown after 2 seconds.".format(idx))
    time.sleep(2)


if __name__ == "__main__":
    # run the following code will resize all images in 'datasets' to (224, 224)
    resize_image(TRAIN_DIR, (224, 224))
    resize_image(TEST_DIR, (224, 224))
    resize_image(VAL_DIR, (224, 224))
    '''
    # comment: use this to test if you successfully resize image or not.

    img = cv2.imread(os.path.join(TRAIN_DIR, "fer0012156.png"))
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)
    '''
