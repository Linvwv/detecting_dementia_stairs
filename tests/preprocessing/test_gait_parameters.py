import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np

import unittest

from preprocessing import gait_params

class TestGaitParams(unittest.TestCase):
    def test_get_num_steps(self):
        test_cases = [
            (
                {"lf_sensor": np.array([[1, 2, 3], [4, 5, 6]]), "rf_sensor": np.array([[1, 2, 3, 4], [1, 2, 3, 4]])},
                7
            ),
            (
                {"lf_sensor": np.array([[], []]), "rf_sensor": np.array([[], []])},
                0
            ),
            (
                {"lf_sensor": np.array([[], []]), "rf_sensor": np.array([[1,2], [3,4]])},
                2
            )

        ]
        for swing_indices, num_steps in test_cases:
            self.assertEqual(gait_params.get_num_steps(swing_indices), num_steps)

    def test_get_cadence(self):
        test_cases = [
            (
                {"num_steps": 7, "trial_time": 7},
                60
            ),
            (
                {"num_steps": 7, "trial_time": 2},
                3.5 * 60
            ),
            (
                {"num_steps": 0, "trial_time": 7},
                0
            )
        ]
        for params, true_result in test_cases:
            self.assertEqual(true_result, gait_params.get_cadence(**params))

    def test_get_cadence_error(self):
        params = {"num_steps": 5, "trial_time": 0}
        self.assertRaises(ZeroDivisionError, gait_params.get_cadence, **params)
        try:
            gait_params.get_cadence(**params)
        except ZeroDivisionError as e:
            self.assertEqual(str(e), "Trial time cannot be 0")

    def test_get_swing_times(self):
        test_cases = [
            (
                {"foot_swing_indices": np.array([[1, 3.5, 4.9], [2.5, 4, 5.7]]), "frequency": 1},
                np.array([1.5, 0.5, 0.8])
            ),
            (
                {"foot_swing_indices": np.array([[], []]), "frequency": 1},
                np.array([])
            ),
            (
                {"foot_swing_indices": np.array([[50, 72, 91], [65, 89, 123]]), "frequency": 50.0},
                np.array([15.0, 17, 32]) /50.0
            )
        ]
        for params, true_result in test_cases:
            test_result = gait_params.get_swing_times(**params)
            self.assertTrue(np.allclose(test_result, true_result))

    def test_get_stance_times(self):
        test_cases = [
            (
                {"foot_swing_indices": np.array([[1, 3.5, 4.9], [2.5, 4, 5.7]]), "frequency": 1},
                np.array([1, 0.9])
            ),
            (
                {"foot_swing_indices": np.array([[], []]), "frequency": 1},
                np.array([])
            ),
            (
                {"foot_swing_indices": np.array([[50, 72, 91], [65, 89, 123]]), "frequency": 50.0},
                np.array([7, 2]) / 50.0
            )
        ]
        for params, true_result in test_cases:
            test_result = gait_params.get_stance_times(**params)
            self.assertTrue(np.allclose(test_result, true_result))

if __name__ == '__main__':
    unittest.main()
