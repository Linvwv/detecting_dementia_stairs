import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import pandas as pd
import numpy as np

import unittest
from unittest.mock import MagicMock, mock_open, patch

from input.data_file import DataFile

class TestDataFile(unittest.TestCase):
    def test_get_frequency(self):
        test_cases = [
            (
                "// Start Time: Unknown\n// Update Rate: 50.0Hz\n// Filter Profile: human (34.3)\n// Firmware Version: 2.0.1",
                50.0
            ),
            (
                "// Start Time: Unknown\n// Update Rate: 73.52Hz\n// Filter Profile: human (34.3)\n// Firmware Version: 2.0.1",
                73.52
            ),
            (
                "// Update Rate: 73.52Hz\n",
                73.52
            )
        ]

        for read_data, correct_value in test_cases:
            with patch("builtins.open", mock_open(read_data=read_data)):
                file_path = "some/file/path"

                data_file = DataFile(file_path)
                frequency = data_file.get_frequency()

                self.assertEqual(frequency, correct_value)
                self.assertEqual(type(frequency), float)

    def test_get_data_from_file(self):
        pd.read_table = MagicMock()

        test_cases = [
            (
                pd.DataFrame({"Gyr_X": [1, 2, 3], "Gyr_Y": [4, 5, 6], "Gyr_Z": [7, 8, 9],
                              "Acc_X": [10, 11, 12], "Acc_Y": [13, 14, 15], "Acc_Z": [16, 17, 18],
                              "Roll": [19, 20, 21], "Pitch": [22, 23, 24], "Yaw": [25, 26, 27]
                              }),
                {
                    'Angles': np.array([[19, 22, 25], [20, 23, 26], [21, 24, 27]]),
                    'Acc': np.array([[10, 13, 16], [11, 14, 17], [12, 15, 18]]),
                    'Gyr': np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
                },
                {"attributes": ["Gyr", "Acc", "Angles"]}
            ),
            (
                pd.DataFrame({"Gyr_X": [1, 2, 3], "Gyr_Y": [4, 5, 6], "Gyr_Z": [7, 8, 9],
                              "1d_attr": [10,11,12]
                              }),
                {
                    'Gyr': np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
                    '1d_attr': np.array([10,11,12])
                },
                {"attributes": ["Gyr", "1d_attr"]}
            ),
            (
                pd.DataFrame({"1d_attr": [10.1234566, 11.1234561, 12.1234565]}),
                {
                    '1d_attr': np.array([10.123457, 11.123456, 12.123456])
                },
                {"attributes": ["1d_attr"]}
            )
        ]
        data_file = DataFile("some/file/path")

        for check_val, correct_value, params in test_cases:
            pd.read_table.return_value = check_val
            data = data_file.get_data_from_file(**params)

            for key in correct_value:
                self.assertTrue(np.array_equal(correct_value[key],data[key]))

if __name__ == '__main__':
    unittest.main()