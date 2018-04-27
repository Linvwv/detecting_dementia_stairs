"""
This file contains utilities for conducting the preprocesing - ie for calculating the features used in this system.
"""


import numpy as np
from sklearn.utils import resample
import pandas as pd

import settings


def clip_readings(trial_data, start_index, trial_time, frequency=50.0, buffer=25, get_trial_length=False):
    """
    Clips the readings retrieved from the data file so that they only contain the data actually generated during the
    trial. It adds a short buffer on either end of the data being clipped to allow for a margin of error. There is also
    error handling for the cases in which the start or end indices of a trial are before the start of the trial or after
    the end of the trial. In general this is done by trusting the trial_time over the start indices identified - ie if
    the end index identified goes past the last reading then set it to be the last reading and adjust the start index in
    such a way that the trial time is still maintained. This was done as the start indices were identified from noisy
    plots of the data where as trial times were recorded.
    :param trial_data: The data of a trial. A dictionary mapping from the names of attributes to a sub-dictionary in
                       turn mapping from the names of sensors to the data for that attribute for that sensor
    :param start_index: The start index of the trial - the start of the first
    :param trial_time: The total time of the trial in seconds
    :param frequency: The frequency at which the trial was recorded
    :param buffer: The size of the buffer to add on either side
    :param get_trial_length: A flag that allows the method to return the trial length as well as the clipped trial data.
                             This is used in the RNN models that need to know where the trial ends. This length is the
                             number of frames not time.
    :return: The trial data and potential the trial lengths (if the get_trial_length flag is set to true)
    """
    trial_frames = int(np.ceil(trial_time * frequency))

    start_index = start_index - buffer
    end_index = start_index + trial_frames + buffer

    for attribute in trial_data:
        for sensor in trial_data[attribute]:
            if end_index > len(trial_data[attribute][sensor]):
                end_index = len(trial_data[attribute][sensor])
                start_index = end_index - trial_frames - (buffer * 2)
            if start_index < 0:
                start_index = 0
                end_index = trial_frames + (buffer * 2)
            trial_data[attribute][sensor] = trial_data[attribute][sensor][int(start_index):int(end_index)]

    if get_trial_length:
        return trial_data, len(trial_data[attribute][sensor])
    else:
        return trial_data


def standardise(X):
    """
        Standardises the features by subtracting the mean of each feature from the values of that feature and dividing
        by the standard deviation
        :param X - a numpy array of the features, shape - (data_points, features)
        :returns standardised_X - a numpy array of the standardised features, shape - (data_points, features)
    """
    avg = np.mean(X, axis=0)
    std_dev = np.sqrt(np.var(X, axis=0))

    return (X - avg) / std_dev


def up_sample(X, y, sequence_lengths=None):
    """
        Up samples a data set (given by features X and labels y) so that the data set contains an equal number of data
        points with each label. It does this by resampling with replacemenet from the data points with the lower number
        of data points
        :param X: a pandas dataframe of the features, shape - (data_points, features)
        :param y: the labels of these features, shape - (data_points, )
        :param sequence_lengths: the number of frames in the trial, shape (data_points, )
        :return: X_upsampled, y_upsampled, (sequence_lengths) - same type and len(shape) as X, y sequence_lengths is
                                                                returned if this is not set to None)
    """
    unique, counts = np.unique(y, return_counts=True)
    type_counts = dict(zip(unique, counts))

    max_count = max(type_counts.values())

    # accumulators for the the X and y (and lengths) values after upsampling is conducted.
    y_upsampled, X_upsampled = y.copy(), X.copy()
    if sequence_lengths is not None:
        lengths_upsampled = sequence_lengths.copy()
    for type in type_counts:
        # for each label (type) randomly sample all lists the following number of times - the difference between the
        # number of occurences of the label with the most samples and the number of occurences of this label that
        # already exist. Add these samples to the dataset.
        type_flags = y == type
        if sequence_lengths is None:
            X_to_add, y_to_add = resample(X[type_flags], y[type_flags], replace=True, n_samples=max_count - len(y[type_flags]))
        else:
            X_to_add, y_to_add, lengths_to_add = resample(X[type_flags], y[type_flags], lengths_upsampled[type_flags], replace=True, n_samples=max_count - len(y[type_flags]))
            lengths_upsampled = pd.concat([lengths_upsampled, lengths_to_add])
        X_upsampled = pd.concat([X_upsampled, X_to_add])
        y_upsampled = pd.concat([y_upsampled, y_to_add])

    # reset the indices after upsampling
    X_upsampled = X_upsampled.reset_index(drop=True)
    y_upsampled = y_upsampled.reset_index(drop=True)
    if sequence_lengths is not None:
        lengths_upsampled = lengths_upsampled.reset_index(drop=True)
        return X_upsampled, y_upsampled, lengths_upsampled
    else:
        return X_upsampled, y_upsampled


def get_resultant(data_attribute):
    """
        Get the resultant of n points in m dimensional space
        :param data_attribute - a list or numpy array, shape - (n, m)
        :returns resultant_data_attribute - a numpy array, shape - (n, ), the norm2 of the vector, calculated as follows
                    sqrt(point[0] ** 2 + point[1] ** 2 + .. point[m]) for point in data_attribute
    """
    if len(data_attribute.shape) != 2:
        raise ValueError("Must be at least 2 dimensional data")
    return np.sqrt(np.sum(np.square(data_attribute), axis=1))


def get_windowed_avgs(data_attribute, num_windows=None):
    """
        Gets windowed average of a potentially multi-dimensional data_attribute of a trial. It divides the data into
        intervals with an equal number of points in each interval and returns the mean of each interval. If
        multidimensional data is used then the mean will also be multidimensional. If the number of windows is greater
        than the number of points in the data attribute then zeros are padded to the end to maintain the size.
        :param data_attribute - a list or numpy array of n points in m dimensions, shape - (n, m)
        :param num_windows - int, the number of intervals to divide the data_attribute into.
        :returns interval_mean - a list of the mean data points of the intervals, shape - (interval_num, m)
    """
    if num_windows is None:
        num_windows = settings.num_windows

    if num_windows > len(data_attribute):
        # if the number of data points is less than the number of windows then pad the end of the data with zeros to
        # maintain a constant size.
        padded_shape = (num_windows) if len(data_attribute.shape) == 1 else (num_windows, data_attribute.shape[1])
        ret = np.zeros(padded_shape, dtype=data_attribute.dtype)
        if type(padded_shape) is int:
            ret[:data_attribute.shape[0]] = data_attribute
        else:
            ret[:data_attribute.shape[0], :data_attribute.shape[1]] = data_attribute
        return list(ret)
    else:
        # calculate the windowed average of the inervals.
        interval_length = int(np.floor(data_attribute.shape[0] / num_windows))

        return [
            np.mean(data_attribute[i * interval_length: (i + 1) * interval_length], axis=0) if i < num_windows - 1
            else np.mean(data_attribute[i * interval_length:], axis=0)
            for i in range(num_windows)
        ]


def get_swing_indices(data_list, threshold=0.8):
    """
        Calculates the swing_indices - the start and end of a swing phase of a leg - based on some 1 dimensional list
        of data and comparing each value to some threshold, if the value is greater than the threshold then that point
        is part of a swing phase. However, to prevent random noise from affecting this there must be 5 samples
        consecutive samples in a swing phase. In this way if the data attribute is noisey and randomly goes over the
        threshold for a short amount of time then this won't effect the swing indices calculated.
        :param data_list - a list of 1 dimensional data from which to calculate the start and end of the swing phases.
        :param threshold - the threshold that points in the data_list must be over in order to be considered in swing.
        :returns swing_indices - a list of 2 lists, the 1st being the start indices of the swing phase and the 2nd
                                 being the end indices
    """
    # find all points where the value is less than threshold
    rest_indices = [i for i in range(len(data_list)) if data_list[i] < threshold]

    swing_indices = [[], []]
    # set swing indices only if there are 5 consecutive samples for which the foot is not at rest
    for i in range(len(rest_indices) - 1):
        if rest_indices[i + 1] - rest_indices[i] > 5:
            swing_indices[0].append(rest_indices[i])
            swing_indices[1].append(rest_indices[i + 1])

    return swing_indices