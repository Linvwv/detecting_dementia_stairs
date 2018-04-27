"""
This class provides the main entry point into the system and provides code for running the various models and retrieving
the various analyses made. It has been set up to demonstrate the system created.
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import settings
from analysis.classical_model import ClassicalModel
from analysis.rnn_model import RNNModel
from analysis.splitters import LeaveOneOutSplitter, RandomSplitter
from input.dataset import DataSet
from preprocessing import feature_extraction
from analysis import runners

# SETUP
# the number of windows. For classical models the best was found to be 8. For rnn models use settings.max_reading_length
settings.num_windows = 8
# the sensors to be used
settings.used_sensors = settings.foot_sensors
# the sensor attributes to be used
settings.used_sensor_attributes = ["Acc", "Gyr"]
# the data directory to be used
settings.data_dir = "data/"
# the name of the look up file
settings.lookup_file = 'SWTS1_Xsens_Look_Up.xlsx'

# A function that maps the label from 3 classes to 2 classes
label_mapper = lambda y: y > 0
# The name of the label trying to be predicted in the dataset
label_name = "Group"

# THE DIFFERENT CLASSIFIER PARAMETERS FOR THE CLASSICAL MODELS - choose one to run analyses on
# Gradient Boosting Classifier
clf_type = GradientBoostingClassifier
# binary params
clf_params_bin = {'min_samples_leaf': 2, 'learning_rate': 0.15, 'n_estimators': 80, 'max_depth': 1}
# multiclass params
clf_params_mult = {'learning_rate': 0.05, 'n_estimators': 70, 'max_depth': 3, 'min_samples_leaf': 2}
# Human readable name of the model
model_name = "Gradient Boosting Classifier"

# # Random Forest Classifier
# clf_type = RandomForestClassifier
# # binary params
# clf_params_bin = {'min_samples_leaf': 5, 'n_estimators': 120, 'max_depth': None}
# # multiclass params
# clf_params_mult = {'n_estimators': 120, 'max_depth': 3, 'min_samples_leaf': 2}
# # Human readable name of the model
# model_name = "Random Forest Classifier"

# # MLP Classifier
# clf_type = MLPClassifier
# # binary params
# clf_params_bin = {'hidden_layer_sizes': (8,), 'learning_rate_init': 0.0002, 'max_iter': 750, 'activation': 'logistic'}
# # multiclass params
# clf_params_mult = {'hidden_layer_sizes': (8,), 'learning_rate_init': 0.0002, 'max_iter': 750, 'activation': 'logistic'}
# Human readable name of the model
# model_name = "MLP Classifier"

# THE DIFFERENT CLASSIFIER PARAMETERS FOR THE CLASSICAL MODELS - choose one to run analyses on
# # The parameters for the binary LSTM RNN
# clf_params_bin = {
#     "rnn_cell_type": tf.contrib.rnn.LSTMCell,
#     "num_stacks": 2,
#     "fully_connected_cells": 8,
#     "num_rnn_units": 64,
#     "learning_rate": 0.001
# }
# # The parameters for the multiclass LSTM RNN
# clf_params_mult = {
#     "rnn_cell_type": tf.contrib.rnn.LSTMCell,
#     "num_stacks": 4,
#     "fully_connected_cells": 8,
#     "num_rnn_units": 64,
#     "learning_rate": 0.001
# }
# The parameters for the binary GRU RNN
# clf_params_bin = {
#     "rnn_cell_type": tf.contrib.rnn.GRUCell,
#     "num_stacks": 2,
#     "fully_connected_cells": 8,
#     "num_rnn_units": 128,
#     "learning_rate": 0.001
# }
# # The parameters for the multiclass GRU RNN
# clf_params_mult = {
#     "rnn_cell_type": tf.contrib.rnn.GRUCell,
#     "num_stacks": 2,
#     "fully_connected_cells": 8,
#     "num_rnn_units": 128,
#     "learning_rate": 0.001
# }

# Read and setup the dataset
dataset = DataSet()
dataset.setup(ignore_trials=settings.ignore_trials)

# Calculate the features
feature_names = feature_extraction.calc_features(dataset, get_lengths=True)

# Binary Classical Model
model = ClassicalModel(2, clf_type, clf_params=clf_params_bin, label_mapper=label_mapper, upsampling=True)
# Multiclass Classical Model
# model = ClassicalModel(3, clf_type, clf_params=clf_params_bin, label_mapper=None, upsampling=True)

# Binary RNN Model
# model = RNNModel(2, len(feature_names), label_mapper=label_mapper, clf_params=clf_params_bin, num_epochs=20, upsampling=True)
# Multiclass RNN Model
# model = RNNModel(3, len(feature_names), label_mapper=None, clf_params=clf_params_bin, num_epochs=20, upsampling=True)

# The splitter - Leave One (whole participant) Out splitter
splitter = LeaveOneOutSplitter(split_param_name="Participant code", split_param_values=dataset.get_data_attribute("Participant code").unique())

# Print the model being used
if type(model) is ClassicalModel:
    print(model.clf_type, model.clf_params)
else:
    print("RNN", model.clf_params)

# Run the model across all splits and get the accuracy
train_accuracy, test_accuracy = runners.get_total_score_all_splits(dataset.data, feature_names, label_name, model, splitter)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy, "\n")
splitter.reset()

# # Run the model across all splits and get the confusion matrix for each split
# train_conf_mats, test_conf_mats = runners.get_property_for_splits(dataset.data, feature_names, label_name, model, splitter, property_getter=model.get_train_test_confusion_matrix)
# total_train_conf_mats = np.sum(train_conf_mats, axis=0)
# total_test_conf_mats = np.sum(test_conf_mats, axis=0)
# print("Train confusion Matrix:\n", total_train_conf_mats)
# print("Test confusion Matrix:\n", total_test_conf_mats, "\n")
# splitter.reset()
#
# # Run the model across all splits and get the average learning curve
# runners.plot_avg_learning_curve_all_splits(dataset.data, feature_names, label_name, model, splitter, model_name)
# splitter.reset()
#
# # Run the model across all splits and plot the average feature importances only works for Gradient Boosting Classifier
# # and Random Forest Classifier
# runners.plot_feature_importances(dataset.data, feature_names, label_name, model, splitter, model_name, show_var=False)
# splitter.reset()