import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np
import pandas as pd

import unittest
from unittest.mock import MagicMock

from preprocessing import feature_extraction, preprocess
import settings
from input import input_utils

feature_frame = None
class TestFeatureExtraction(unittest.TestCase):
    def test_calc_attr(self):
        settings.used_sensors = ("lf_sensor", "rf_sensor")
        test_cases = [
            (
                # attribute calculation function
                lambda x: sum(x),
                # data on which to calculate attribute
                {"lf_sensor": [1, 2, 3], "rf_sensor": [4, 5, 6], "p_sensor": [7, 8, 9]},
                # global parameter
                None,
                # sensors
                None,
                # return list flag
                False,
                # true return value
                {"lf_sensor": 6, "rf_sensor": 15}
            ),
            (
                # attribute calculation function
                lambda x, y: sum(x) + y,
                # data on which to calculate attribute
                {"lf_sensor": [1, 2, 3], "rf_sensor": [4, 5, 6], "p_sensor": [7, 8, 9]},
                # global parameter
                5,
                # sensors
                None,
                # return list flag
                False,
                # true return value
                {"lf_sensor": 11, "rf_sensor": 20}
            ),
            (
                # attribute calculation function
                lambda x, y: sum(x) + y,
                # data on which to calculate attribute
                {2: [1, 2, 3], 1: [4, 5, 6]},
                # global parameter
                5,
                # sensors
                (1, 2),
                # return list flag
                False,
                # true return value
                {2: 11, 1: 20}
            ),
            (
                # attribute calculation function
                lambda x, y: [sum(x) + y],
                # data on which to calculate attribute
                {2: [1, 2, 3], 1: [4, 5, 6]},
                # global parameter
                5,
                # sensors
                (1, 2),
                # return list flag
                True,
                # true return value
                [20, 11]
            ),
        ]

        for attr_getter_func, data, glob_param, sensors, return_list, true_return_val in test_cases:
            test_val = feature_extraction.calc_attr(attr_getter_func, data, glob_param=glob_param, sensors=sensors, return_list=return_list)
            if type(true_return_val) is dict:
                self.assertDictEqual(test_val, true_return_val)
            else:
                self.assertListEqual(test_val, true_return_val)

    def test_get_raw_data_features(self):
        settings.num_windows = 1
        test_cases = [
            (
                # data for calculating features
                {"x": {"lf_sensor": [1,2,3]}, "y": {"lf_sensor": [4,5,6]}},
                # data attributes used to calculate features
                ("x", "y"),
                # sensors
                ("lf_sensor", ),
                # get feature names flag
                False,
                # calc attr mock - what to return from the calc_attr method
                ["some_feature"],
                # true result
                ["some_feature", "some_feature"]
            ),
            (
                # data for calculating features
                {"x": {"lf_sensor": [1, 2, 3], "rf_sensor": [4,5,6]}, "y": {"lf_sensor": [4, 5, 6], "rf_senosr": [7,8,9]}},
                # data attributes used to calculate features
                ("x", "y"),
                # sensors
                ("lf_sensor", "rf_sensor"),
                # get feature names flag
                True,
                # calc attr mock - what to return from the calc_attr method
                ["lf_some_feature", "rf_some_feature"],
                # true result
                (["lf_some_feature", "rf_some_feature", "lf_some_feature", "rf_some_feature"], ['lf_sensor_res_x_window0', 'rf_sensor_res_x_window0', 'lf_sensor_res_y_window0', 'rf_sensor_res_y_window0'])
            )
        ]
        for data, attributes, sensors, get_feature_names, calc_attr_mock, true_result in test_cases:
            feature_extraction.calc_attr = MagicMock()
            feature_extraction.calc_attr.return_value = calc_attr_mock
            test_result = feature_extraction.get_raw_data_features(data, attributes, sensors, get_feature_names)
            if get_feature_names:
                self.assertTupleEqual(test_result, true_result)
            else:
                self.assertListEqual(test_result, true_result)

    def test_get_trial_features(self):
        settings.num_windows = 1
        test_cases = [
            (
                # data for calculating features
                {"x": {"lf_sensor": [1,2,3]}, "y": {"lf_sensor": [4,5,6]}},
                # data attributes used to calculate features
                ("x", "y"),
                # sensors
                ("lf_sensor", ),
                # get feature names flag
                False,
                # calc attr mock - what to return from the calc_attr method
                ["some_feature"],
                # true result
                ["some_feature", "some_feature"]
            ),
            (
                # data for calculating features
                {"x": {"lf_sensor": [1, 2, 3], "rf_sensor": [4,5,6]}, "y": {"lf_sensor": [4, 5, 6], "rf_senosr": [7,8,9]}},
                # data attributes used to calculate features
                ("x", "y"),
                # sensors
                ("lf_sensor", "rf_sensor"),
                # get feature names flag
                True,
                # calc attr mock - what to return from the calc_attr method
                ["lf_some_feature", "rf_some_feature"],
                # true result
                (["lf_some_feature", "rf_some_feature", "lf_some_feature", "rf_some_feature"], ['lf_sensor_res_x_window0', 'rf_sensor_res_x_window0', 'lf_sensor_res_y_window0', 'rf_sensor_res_y_window0'])
            )
        ]
        for data, attributes, sensors, get_feature_names, calc_attr_mock, true_result in test_cases:
            feature_extraction.calc_attr = MagicMock()
            feature_extraction.calc_attr.return_value = calc_attr_mock
            test_result = feature_extraction.get_trial_features(data, attributes, sensors, get_feature_names)
            if get_feature_names:
                self.assertTupleEqual(test_result, true_result)
            else:
                self.assertListEqual(test_result, true_result)

    def test_calc_features(self):
        global feature_frame
        keep_func = feature_extraction.get_trial_features
        def side_effect(arg):
            global feature_frame
            feature_frame = arg

        test_cases = [
            (
                # num trials
                3,
                # start times
                [1, 2, 0],
                # trial_times
                np.array([2000, 3000, 3000]),
                # trial data
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7]
                ],
                # clipped trial data
                [
                    [2, 3],
                    [4, 5, 6],
                    [3, 4, 5]
                ],
                # trial features - these are used to mock the get_trial_features method
                [
                    ([1, 2], ["x_feat", "y_feat"]),
                    [3, 4],
                    [5, 6]
                ],
                # true feature frame
                pd.DataFrame(
                    {
                        "x_feat": [1, 3, 5],
                        "y_feat": [2, 4, 6]
                    }
                ),
                # feature names
                np.array(["x_feat", "y_feat"]),
                # get sequence lengths
                False
            ),
            (
                # num trials
                3,
                # start times
                [1, 2, 0],
                # trial_times
                np.array([2000, 3000, 3000]),
                # trial data
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7]
                ],
                # clipped trial data
                [
                    ([2, 3], 2),
                    ([4, 5, 6], 3),
                    ([3, 4, 5], 3)
                ],
                # trial features - these are used to mock the get_trial_features method
                [
                    ([1, 2], ["x_feat", "y_feat"]),
                    [3, 4],
                    [5, 6]
                ],
                # true feature frame
                pd.DataFrame(
                    {
                        "x_feat": [1, 3, 5],
                        "y_feat": [2, 4, 6],
                    }
                ),
                # feature names
                np.array(["x_feat", "y_feat"]),
                # get sequence lengths
                False
            )
        ]
        for num_trials, start_times, trial_times, trial_data, clipped_trial_data, trial_features, true_feat_frame, \
            true_feat_names, get_seq_lengths in test_cases:
            feature_frame = None
            dataset = MagicMock()

            dataset.get_num_trials.return_value = num_trials
            dataset.get_data_attribute.side_effect = [start_times, trial_times]
            dataset.set_data_attributes.side_effect = side_effect
            dataset.data = {}

            input_utils.read_trial_data = MagicMock()
            input_utils.read_trial_data.side_effect = trial_data

            preprocess.clip_readings = MagicMock()
            preprocess.clip_readings.side_effect = clipped_trial_data

            feature_extraction.get_trial_features = MagicMock()
            feature_extraction.get_trial_features.side_effect = trial_features

            preprocess.standardise = MagicMock()
            preprocess.standardise.side_effect = lambda x: x

            feature_names = feature_extraction.calc_features(dataset, get_seq_lengths)

            self.assertTrue(true_feat_frame.equals(feature_frame))
            if get_seq_lengths is True:
                self.assertListEqual(dataset.data["sequence_lengths"], [x[1] for x in clipped_trial_data])
                self.assertTrue(np.array_equal(feature_names[0], true_feat_names))
            else:
                self.assertTrue(np.array_equal(feature_names, true_feat_names))

        input_utils.read_trial_data.side_effect = None
        dataset.set_attributes.side_effect = None
        dataset.get_data_attribute.side_effect = None
        preprocess.clip_readings.side_effect = None
        feature_extraction.get_trial_features.side_effect = None
        preprocess.standardise.side_effect = None
        feature_extraction.get_trial_features = keep_func

if __name__ == '__main__':
    unittest.main()