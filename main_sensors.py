"""
This file contains the code that was used to analyse the importance of each sensor in making a prediction. This is
conducted across the Gradient Boosting Classifier and the Random Forest Classifier for both the binary and multiclass
settings
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import settings
from analysis.splitters import LeaveOneOutSplitter
from input.dataset import DataSet
from preprocessing import feature_extraction
from analysis.classical_model import ClassicalModel
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

def all_sensor_perms(model):
    """
    Given a model calculate and print the accuracy on the accuracy on the exhaustive cross validation set for each
    permutation of sensors.
    :param model: The model to be used for the analysis
    """
    sensor_combinations = [
        ["lf_sensor"],
        ["rf_sensor"],
        ["p_sensor"],
        ["lf_sensor", "rf_sensor"],
        ["lf_sensor", "p_sensor"],
        ["rf_sensor", "p_sensor"],
        ["lf_sensor", "rf_sensor", "p_sensor"]
    ]
    for used_sensors in sensor_combinations:
        settings.used_sensors = used_sensors
        dataset = DataSet()
        dataset.setup(ignore_trials=settings.ignore_trials)
        splitter = LeaveOneOutSplitter(split_param_name="Participant code",
                                            split_param_values=dataset.get_data_attribute("Participant code").unique())

        feature_names = np.array(feature_extraction.calc_features(dataset))

        cum_score = 0
        for j in range(100):
            cum_score += \
            runners.get_total_score_all_splits(dataset.data, feature_names, label_name, model, splitter)[1]
            splitter.reset()
        print("Sensors", settings.used_sensors, "Score", cum_score / 100)

# The name of the label in the dataset
label_name = "Group"
# The label mappers to be used
label_maps = [lambda y: y > 0, None]

# The various classifiers and their parameters
clfs = [
        (
            GradientBoostingClassifier,
            {'min_samples_leaf': 2, 'learning_rate': 0.15, 'n_estimators': 80, 'max_depth': 1}
        ),
        (
            RandomForestClassifier,
            {'min_samples_leaf': 5, 'n_estimators': 120, 'max_depth': None}
        )
]

for label_map in label_maps:
    print("Binary" if label_map else "Multi-Class",  "Results")
    print("------------------------------------------------------------------------------------------------------------------------")

    for clf_type, clf_params in clfs:
        print("Model:", clf_type, clf_params)
        clf = ClassicalModel(2 if label_map else 3, clf_type, label_mapper=label_map, upsampling=True, clf_params=clf_params)

        all_sensor_perms(clf)

        print()