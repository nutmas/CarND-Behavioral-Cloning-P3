import csv  # for csv file import
import numpy as np
import os
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def get_file_data(file_path, header=False):
    # function to read in data from driving_log.csv

    lines = []
    with open(file_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # if header is set to true then skip first line of csv
        if header:
            # if header exists iterate to next item in list, returns -1 if exhausted
            next(reader, -1)
        for line in reader:
            # loop through reader appending each line to lines array
            lines.append(line)
    return lines


def prepare_data(dataset):

    # create 2 arrays to hold images and steering angles
    steering_angles = []
    camera_images = []

    # set array address as meaningful name - to avoid mixup of left, right and centre images
    centre_camera = 0
    left_camera = 1
    right_camera = 2
    steering_angle = 3

    # set location of images
    source_path = './my_driving/IMG/'

    for line in dataset[0:]:

        # get steering angle from csv 4th element of CSV and cast as float for this point in time
        steering_centre = float(line[steering_angle])

        # create adjusted steering measurements for the side camera images
        correction = 0.25  # this is a parameter to tune
        steering_left = steering_centre + correction
        steering_right = steering_centre - correction

        # read in image from file location and image name from cell in csv
        img_centre = cv2.imread(source_path + line[centre_camera].split('/')[-1])
        img_left = cv2.imread(source_path + line[left_camera].split('/')[-1])
        img_right = cv2.imread(source_path + line[right_camera].split('/')[-1])

        #img_centre_hsv = cv2.cvtColor(img_centre, cv2.COLOR_BGR2HSV)
        #img_left_hsv = cv2.cvtColor(img_left, cv2.COLOR_BGR2HSV)
        #img_right_hsv = cv2.cvtColor(img_right, cv2.COLOR_BGR2HSV)

        #img_centre_yuv = cv2.cvtColor(img_centre, cv2.COLOR_BGR2YUV)
        #img_left_yuv = cv2.cvtColor(img_left, cv2.COLOR_BGR2YUV)
        #img_right_yuv = cv2.cvtColor(img_right, cv2.COLOR_BGR2YUV)

        # add images and angles to data set
        camera_images.append(img_centre)
        camera_images.append(img_left)
        camera_images.append(img_right)

        steering_angles.append(steering_centre)
        steering_angles.append(steering_left)
        steering_angles.append(steering_right)

    # convert to numpy arrays for keras
    X_train = np.array(camera_images)
    y_train = np.array(steering_angles)

    return X_train, y_train


def net_NVIDIA():
    # NVIDIA Convolutional Network function

    # create a sequential model
    model = Sequential()

    # add pre-processing steps - normalising the data and mean centre the data
    # add a lambda layer for normalisation
    # normalise image by divide each element by 255 (max value of an image pixel)
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # after image is normalised in a range 0 to 1 - mean centre it by subtracting 0.5 from each element - shifts mean from 0.5 to 0
    # training loss and validation loss should be much smaller

    # crop the image to remove pixels that are not adding value - top 70, and bottom 25 rows
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # keras auto infer shape of all layers after 1st layer
    # 1st layer
    #model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation="relu"))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    # 2nd layer
    #model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    # 3rd layer
    #model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    # 4th layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # 5th layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # 6th layer
    model.add(Flatten())
    # 7th layer - add fully connected layer ouput of 100
    model.add(Dense(100))
    # 8th layer - add fully connected layer ouput of 50
    model.add(Dense(50))
    # 9th layer - add fully connected layer ouput of 10
    model.add(Dense(10))
    # 0th layer - add fully connected layer ouput of 1
    model.add(Dense(1))

    # summarise model output on screen
    model.summary()

    return model


def train_model(model, inputs, outputs, model_path, set_epochs=3):

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # split training data out as 20% for validation and traing model
    history_object = model.fit(inputs, outputs, validation_split=0.2, shuffle=True, epochs=set_epochs, verbose=1)

    # save model output with name of net and number of epochs as name
    model_object = model_path + 'Final' + str(set_epochs) + '.h5'
    model.save(model_object)
    print("Model saved at " + model_object)

    return history_object


if __name__ == '__main__':

    # load the data set from data location
    dataset = get_file_data('./my_driving')

    # create a variable to hold images; X_train
    # and variable to hold steerign angles y_train
    X_train, y_train = prepare_data(dataset)

    # create an instance of an NVIDIA model
    model = net_NVIDIA()
    # set number of epoch for training
    num_epoch = 6

    # train the model
    history_object = train_model(model, X_train, y_train, './NVidia_', num_epoch)

    print("Model training complete")
