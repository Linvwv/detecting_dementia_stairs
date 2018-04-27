import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np
import pandas as pd

import unittest

from preprocessing import preprocess

class TestPreProcess(unittest.TestCase):
    def test_standardise(self):
        test_cases = [
            (
                np.array([[1,2,3,4],
                          [5,6,7,8],
                          [9,10,11,12]]),
                np.array([[-1.22474487, -1.22474487, -1.22474487, -1.22474487],
                          [0., 0., 0.,  0.],
                          [1.22474487, 1.22474487, 1.22474487, 1.22474487]])
            ),
            (
                np.array([[1.592], [800.528], [-40392.48943]]),
                np.array([[0.6862352], [0.72777499], [-1.41401019]])
            ),
        ]
        for x, std_x in test_cases:
            self.assertTrue(np.allclose(preprocess.standardise(x), std_x))

    def test_get_resultant(self):
        test_cases = [
            (
                np.array([[4, 3, 9], [6,8,10], [4,2,500], [-4,2818,3472]]),
                np.array([10.295630141, 14.1421356237, 500.0199996, 4471.6802211249])
            ),

        ]
        for data, res_data in test_cases:
            self.assertTrue(np.allclose(preprocess.get_resultant(data), res_data))

    def test_get_resultant_error(self):
        data = np.array([1,2,3])
        self.assertRaises(ValueError, preprocess.get_resultant, data)
        try:
            preprocess.get_resultant(data)
        except ValueError as e:
            self.assertEqual(str(e), "Must be at least 2 dimensional data")

    def test_get_swing_indices(self):
        test_cases = [
            (
                {"data_list": np.array([])},
                np.array([[], []])
            ),
            (
                {"data_list": np.array([0, 0.4, 0.99999, 1, 2, 304, 2, 1, 0.999999]), "threshold": 1},
                np.array([[2], [8]])
            ),
            (
                {"data_list": np.array([0, 0.4, 0.99999, 1, 2, 304, 2, 1, 0.999999, 0, 0.9999, 1, 1.1, 2, 0]), "threshold": 1},
                np.array([[2], [8]])
            ),
            (
                {"data_list": np.array([0, 0.4, 0.99999, 1, 2, 304, 2, 1, 0.999999, 0, 0.9999, 1, 1.1, 2, 3, 4, 5, 0]),
                 "threshold": 1},
                np.array([[2, 10], [8, 17]])
            )
        ]
        for params, swing_indices in test_cases:
            self.assertTrue(np.array_equal(preprocess.get_swing_indices(**params), swing_indices))

    def test_get_interval_means(self):
        test_cases = [
            (
                {"data_attribute": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), "num_windows": 2},
                np.array([[2.5, 3.5, 4.5], [8.5, 9.5, 10.5]])
            ),
            (
                {"data_attribute": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), "num_windows": 5},
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [0, 0, 0]])
            ),
            (
                {"data_attribute": np.array([1, 5, 6, 8, 2, -8, 4, -6, 2, 9]), "num_windows": 4},
                np.array([3, 7, -3, 2.25])
            ),
            (
                {"data_attribute": np.array([1, 5, 6, 8, 2, -8, 4, -6, 2, 9]), "num_windows": 15},
                np.array([1, 5, 6, 8, 2, -8, 4, -6, 2, 9, 0, 0, 0, 0, 0])
            )
        ]
        for params, result in test_cases:
            self.assertTrue(np.array_equal(preprocess.get_windowed_avgs(**params), result))

    def test_clip_readings(self):
        test_cases = [
            (
                {
                    "trial_data": {
                        "attr_x": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                        "attr_y": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                    },
                    "start_index": 1,
                    "trial_time": 5 / 3,
                    "frequency": 2,
                    "buffer": 0
                },
                {
                    "attr_x": {"sensor1": [2, 3, 4, 5]},
                    "attr_y": {"sensor1": [2, 3, 4, 5]}
                }
            ),
            (
                {
                    "trial_data": {
                        "attr_x": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sensor2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                        "attr_y": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sensor2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                    },
                    "start_index": 5,
                    "trial_time": 6,
                    "frequency": 1,
                    "buffer": 0
                },
                {
                    "attr_x": {"sensor1": [5, 6, 7, 8, 9, 10], "sensor2": [5, 6, 7, 8, 9, 10]},
                    "attr_y": {"sensor1": [5, 6, 7, 8, 9, 10], "sensor2": [5, 6, 7, 8, 9, 10]}
                }
            ),
            (
                {
                    "trial_data": {
                        "attr_x": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sensor2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                        "attr_y": {"sensor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sensor2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                    },
                    "start_index": 0,
                    "trial_time": 5,
                    "frequency": 1,
                    "buffer": 1
                },
                {
                    "attr_x": {"sensor1": [1, 2, 3, 4, 5, 6, 7], "sensor2": [1, 2, 3, 4, 5, 6, 7]},
                    "attr_y": {"sensor1": [1, 2, 3, 4, 5, 6, 7], "sensor2": [1, 2, 3, 4, 5, 6, 7]}
                }
            ),
            (
                {
                    "trial_data": {
                        "attr_x": {"sensor1": [1, 2, 3, 4, 5], "sensor2": [1, 2, 3, 4, 5]},
                        "attr_y": {"sensor1": [1, 2, 3, 4, 5], "sensor2": [1, 2, 3, 4, 5]}
                    },
                    "start_index": 0,
                    "trial_time": 6,
                    "frequency": 1,
                    "buffer": 1
                },
                {
                    "attr_x": {"sensor1": [1, 2, 3, 4, 5], "sensor2": [1, 2, 3, 4, 5]},
                    "attr_y": {"sensor1": [1, 2, 3, 4, 5], "sensor2": [1, 2, 3, 4, 5]}
                }
            )

        ]
        for params, true_result in test_cases:
            test_result = preprocess.clip_readings(**params)
            self.assertDictEqual(test_result, true_result)

    def test_upsample(self):
        test_cases = [
            (
                {
                    "X": pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 9, 5]}),
                    "y": pd.Series([1, 1, 1, 0])
                },
                (
                    pd.DataFrame({"a": [1, 2, 3, 4, 4, 4], "b": [2, 3, 9, 5, 5, 5]}),
                    pd.Series([1, 1, 1, 0, 0, 0])
                )
            ),
            (
                {
                    "X": pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 9, 5, 8]}),
                    "y": pd.Series([1, 1, 1, 0, 2])
                },
                (
                    pd.DataFrame({"a": [1, 2, 3, 4, 5, 4, 4, 5, 5], "b": [2, 3, 9, 5, 8, 5, 5, 8, 8]}),
                    pd.Series([1, 1, 1, 0, 2, 0, 0, 2, 2])
                )
            ),
            (
                {
                    "X": pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 9, 5, 8, 9]}),
                    "y": pd.Series([1, 1, 0, 0, 2, 2])
                },
                (
                    pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 9, 5, 8, 9]}),
                    pd.Series([1, 1, 0, 0, 2, 2])
                )
            )
        ]
        for params, true_result in test_cases:
            test_result = preprocess.up_sample(**params)
            self.assertTrue(true_result[0].equals(test_result[0]))
            self.assertTrue(true_result[1].equals(test_result[1]))

if __name__ == '__main__':
    unittest.main()
