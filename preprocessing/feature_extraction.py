"""
This file provides methods for retrieving various features from the input data that are useful for conducting further
analysis, along with helper functions that make it easy for this to be done.
"""


import numpy as np
import pandas as pd

from input import input_utils
from preprocessing import gait_params, preprocess
import settings


def calc_attr(attr_getter_func, params, glob_param=None, sensors=None, return_list=False):
    """
    Calculates an attribute (specified by attr_getter_func) of the data provided (in params). This method allows for
    attributes retrieval function to be specified on the sensor level and be applied across all the sensors in the
    current analysis without modifying the attribute getter. It also allows for attributes to be retrieved in a list
    form that is needed by the classifiers or in a more human readable dictionary.
    :param attr_getter_func: A function for getting a certain attribute of some data from a (single) sensor
    :param params: The parameters to be passed to the function each time it is called. This is the data from which that
                   attribute can be calculated. It is specified as a dictionary mapping from the sensor names to the
                   data for that sensor
    :param glob_param: An optional 'global' parameter that is provided to the attr_getter_func every time it is called.
                       Unlike in the case of params, where the relevant value is retrieved for each function call, this
                       parameter allows for the attr_getter_func to accept a parameter that is constant across all
                       sensors.
    :param sensors: The names of the sensors for which to calculate the attribute
    :param return_list: A boolean flag. If true - a list of the attributes is returned in the order that the sensors are
                        specified in the sensors parameter. Else - a dictionary of the attributes is returned mapping
                        from the sensor names provided in sensors to the attribute data.
    :return: The values of the attribute calculated for each sensor.
    """
    if sensors is None:
        sensors = settings.used_sensors
    if return_list:
        ret = []
    else:
        ret = {}
    for sensor in sensors:
        if return_list:
            if glob_param is None:
                attr_value = attr_getter_func(params[sensor])
                ret += attr_value
                # if attr_getter_func is np.mean:
                #     print(attr_getter_func(params[sensor]))
            else:
                ret += attr_getter_func(params[sensor], glob_param)
        else:
            if glob_param is None:
                ret[sensor] = attr_getter_func(params[sensor])
            else:
                ret[sensor] = attr_getter_func(params[sensor], glob_param)
    return ret


def get_raw_data_features(trial_data, attributes, sensors, get_feature_names):
    """
    This function gets all the raw data features for a single trial (not based on gait parameters or dead reckoning).
    This is done using windowed averages. As it is currently implemented the resultant of each raw data attribute is
    calculated and then a windowed average is used to convert these to a fixed size in the case of classical models or
    these are padded with zeros in the case of the RNN models.
    :param trial_data: The data from a single trial specified as a dictionary mapping from the attribute name to a
                       sub-dictionary in turn mapping from sensor names to the data for that attribute for that sensor.
    :param attributes: The names of the attributes to be used in the current analysis
    :param sensors: The names of the sensors to be used in the current analysis
    :param get_feature_names: A flag that allows for feature names to be retrieved along with with the features. This is
                              set to true for the first trial of the dataset.
    :return: If get_feature_names - a tuple with the first item being the calculated raw_data_features as a
             list and the second item being the names of the features in the order they appear in this list. Else only
             the first item of this tuple is returned.
    """
    raw_data_features = []
    if get_feature_names:
        raw_data_feature_names = []

    for attribute in attributes:
        # add windowed resultant attribute as feature
        res_attr = calc_attr(preprocess.get_resultant, trial_data[attribute], sensors=sensors)
        raw_data_features += calc_attr(preprocess.get_windowed_avgs, res_attr, sensors=sensors, return_list=True)

        # # add windowed attribute as feature
        # windowed_attr = calc_attr(preprocess.get_interval_means, trial_data[attribute], sensors=sensors, return_list=True)
        # windowed_attr = list(np.reshape(windowed_attr, (-1)))
        # raw_data_features += windowed_attr

        if get_feature_names:
            # add the names of the features calculated from the resultant data attribute.
            for sensor in sensors:
                raw_data_feature_names.extend((sensor + "_res_" + attribute + "_window" + str(window) for window in range(settings.num_windows)))
            # # add the names of the features calculated from the multidimensional data attribute.
            # for sensor in sensors:
            #     raw_data_feature_names.extend((sensor + "_" + attribute + "_window" + str(window) + "_dim_" + dim for window in range(num_windows) for dim in ["x", "y", "z"]))

    if get_feature_names:
        return raw_data_features, raw_data_feature_names
    else:
        return raw_data_features


def get_trial_features(trial_data, attributes=None, sensors=None, get_feature_names=False):
    """
    A function for getting all the features for a single file. As is currently implemented this is just the
    raw_data_features, however this method was made to allow more types of features to be added. When this was realised
    to be ineffective these features were removed.
    :param trial_data: The data from a single trial specified as a dictionary mapping from the attribute name to a
                       sub-dictionary in turn mapping from sensor names to the data for that attribute for that sensor.
    :param attributes: The names of the attributes to be used in the current analysis
    :param sensors: The names of the sensors to be used in the current analysis
    :param get_feature_names: A flag that allows for feature names to be retrieved along with with the features. This is
                              set to true for the first trial of the dataset.
    :return: If get_feature_names - a tuple with the first item being the calculated raw_data_features as a list and
             the second item being the names of the features in the order they appear in this list. Else only the first
             item of this tuple is returned.
    """
    # if attributes and sensors are not set just use all the attributes and sensors in trial data
    if attributes is None:
        attributes = trial_data.keys()
    if sensors is None:
        sensors = trial_data[list(attributes)[0]].keys()

    if get_feature_names:
        raw_data_features, raw_data_feature_names = get_raw_data_features(trial_data, attributes, sensors, get_feature_names)
        # gait_param_features, gait_param_feature_names = get_gait_param_features(trial_data["Gyr"], trial_time, get_feature_names)
    else:
        raw_data_features = get_raw_data_features(trial_data, attributes, sensors, get_feature_names)
        # gait_param_features = get_gait_param_features(trial_data["Gyr"], trial_time, get_feature_names)

    trial_features = raw_data_features

    if get_feature_names:
        return trial_features, raw_data_feature_names
    else:
        return trial_features


def calc_features(dataset, get_lengths=False):
    """
    Calculates the features of all trials in the dataset and adds the calculated features to the dataset, it also
    returns the names of all the features added so these can easily be retrieved from the dataset later. This method
    iterates over the dataset, locates the file for a single trial, reads the data from each sensor, clips the readings
    so they only contain the data from the actual trial, calculates the features for this trial and then moves on to the
    next data file. In this way only one data file is read into the system at once.
    :param dataset: The dataset object encapsulating the dataset. This is just the lookup file with the setup method
                    called on it in order to remove unnecessary, corrupted or missing data.
    :param get_lengths: A flag that allows the sequence lengths of each trial (the number of frames in a trial) to be
                        added to the dataset. This is not a feature used by any of the models, however, it is needed by
                        the RNN in order to retrieve only the frames that are a part of the trial.
    :return: The names of the features that have been added to the dataset.
    """
    X = []
    num_trials = dataset.get_num_trials()

    start_indices = dataset.get_data_attribute("Start Times")
    trial_times = dataset.get_data_attribute("DISTANCETRIAL") / 1000.0
    trial_lengths = []
    for trial in range(num_trials):
        # print(dataset.data["ROW"][trial])
        trial_data_files = dataset.get_data_files(trial)
        trial_data = input_utils.read_trial_data(trial_data_files, trial_data_files.keys(), settings.used_sensor_attributes)

        if get_lengths:
            trial_data, trial_frames = preprocess.clip_readings(trial_data, start_indices[trial], trial_times[trial], get_trial_length=get_lengths)
            if settings.num_windows >= trial_frames:
                trial_lengths.append(trial_frames)
            else:
                trial_lengths.append(settings.num_windows)
        else:
            trial_data = preprocess.clip_readings(trial_data, start_indices[trial], trial_times[trial], get_trial_length=get_lengths)

        if trial == 0:
            trial_features, feature_names = get_trial_features(trial_data, get_feature_names=True, sensors=settings.used_sensors)
        else:
            trial_features = get_trial_features(trial_data, sensors=settings.used_sensors)

        X.append(trial_features)

    X = preprocess.standardise(X)

    features_frame = pd.DataFrame(X, columns=feature_names)
    dataset.set_data_attributes(features_frame)
    if get_lengths:
        dataset.data["sequence_lengths"] = trial_lengths

    return np.array(feature_names)