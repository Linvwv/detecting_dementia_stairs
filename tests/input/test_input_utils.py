import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np

import unittest
from unittest.mock import MagicMock

from input import input_utils
from input.data_file import DataFile

class TestInputUtils(unittest.TestCase):
    def test_bring_attribute_up(self):
        test_cases = [
            (
                {
                    "x": {"1": 1, "2": 2, "3": 3},
                    "y": {"1": 4, "2": 5, "3": 6},
                },
                {
                    "1": {"x": 1, "y": 4},
                    "2": {"x": 2, "y": 5},
                    "3": {"x": 3, "y": 6},
                }
            ),
            (
                {
                    "x": {"1": 1, "2": 2},
                    "y": {"1": 4, "2": 5, "3": 6},
                    "z": {"1": 7, "2": 8, "3": 9},
                },
                {
                    "1": {"x": 1, "y": 4, "z": 7},
                    "2": {"x": 2, "y": 5, "z": 8},
                    "3": {"y": 6, "z": 9}
                }
            )
        ]


        for input_dict, true_output_dict in test_cases:
            self.assertDictEqual(input_utils.bring_attributes_up(input_dict), true_output_dict)

    def test_read_trial_data(self):
        DataFile.get_frequency = MagicMock()
        DataFile.get_frequency.return_value = 50.0
        DataFile.get_data_from_file = MagicMock()
        DataFile.get_data_from_file.return_value = {"attr1": np.array([7,8,9]), "attr2": np.array([[1,2,3],[4,5,6]])}

        true_value = {'attr2': {'s1': np.array([[1, 2, 3], [4, 5, 6]]), 's2': np.array([[1, 2, 3], [4, 5, 6]])},
                      'attr1': {'s1': np.array([7, 8, 9]), 's2': np.array([7, 8, 9])}}

        trial_data = input_utils.read_trial_data({"s1": "some/file/path", "s2": "some/other/file/path"}, ["s1", "s2"], ["attr1", "attr2"])

        for attr in true_value:
            for sensor in true_value[attr]:
                self.assertTrue(np.array_equal(true_value[attr][sensor], trial_data[attr][sensor]))

if __name__ == '__main__':
    unittest.main()