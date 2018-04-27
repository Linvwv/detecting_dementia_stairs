import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import pandas as pd
import numpy as np
from os import path

import unittest
from unittest.mock import MagicMock

from input.dataset import DataSet


class TestDataSet(unittest.TestCase):
    def test_remove_empty_rows(self):
        # test cases each element is (data_before, data_result, method_parameters)
        test_cases = [
            (
                pd.DataFrame({
                    "attr1": [1, 2, 3, np.nan, np.nan, 6, 7],
                    "attr2": [1, np.nan, 3, 4, np.nan, 6, 7],
                    "attr3": [np.nan, 2, 3, 4, 5, 6, 7]}),
                pd.DataFrame({
                    "attr1": [1.0, 3, 6, 7],
                    "attr2": [1.0, 3, 6, 7],
                    "attr3": [np.nan, 3, 6, 7]}),
                {"remove_rows": [], "check_columns": ["attr1", "attr2"]}
            ),
            (
                pd.DataFrame({
                    "attr1": [np.nan, np.nan],
                    "attr2": [np.nan, np.nan],
                    "attr3": [np.nan, 2,]}),
                pd.DataFrame({
                    "attr1": [],
                    "attr2": [],
                    "attr3": []}),
                {"remove_rows": [], "check_columns": ["attr1", "attr2"]}
            ),
            (
                pd.DataFrame({
                    "attr1": [1, 2, 3, np.nan, np.nan, 6, 7],
                    "attr2": [1, np.nan, 3, 4, np.nan, 6, 7],
                    "attr3": [np.nan, 2, 3, 4, 5, 6, 7]}),
                pd.DataFrame({
                    "attr1": [3.0, 6],
                    "attr2": [3.0, 6],
                    "attr3": [3.0, 6]}),
                {"remove_rows": [0, 6], "check_columns": ["attr1", "attr2"]}
            ),
            (
                pd.DataFrame({
                    "attr1": [np.nan, np.nan],
                    "attr2": [np.nan, np.nan],
                    "attr3": [np.nan, 2, ]}),
                pd.DataFrame({
                    "attr1": [],
                    "attr2": [],
                    "attr3": []}),
                {"remove_rows": [0,1], "check_columns": ["attr1", "attr2"]}
            )
        ]

        for data_frame, result_data_frame, method_params in test_cases:
            data_set = DataSet()
            # Pandas represents empty cell values as NaN
            data_set.data = data_frame
            data_set.remove_problem_trials(**method_params)

            self.assertEqual(data_set.get_num_trials(), result_data_frame.shape[0])
            self.assertTrue(result_data_frame.equals(data_set.data))

    def test_get_attribute(self):
        test_cases = [
            (
                pd.DataFrame({"attr1": [1, 2, 3]}),
                pd.Series([1, 2, 3]),
                {"columns": "attr1"}
            ),
            (
                pd.DataFrame({"attr1": [1, 2, 3]}),
                1,
                {"columns": "attr1", "row": 0}
            ),
            (
                pd.DataFrame({"attr1": [1, 2, 3]}),
                pd.Series([1,2]),
                {"columns": "attr1", "row": [0, 1]}
            ),
            (
                pd.DataFrame({"attr1": [1, 2, 3], "attr2": [1, 2, 3], "attr3": [1, 2, 3]}),
                pd.DataFrame({"attr1": [1, 2, 3], "attr2": [1, 2, 3]}),
                {"columns": ["attr1", "attr2"]}
            )
        ]

        for data_frame, correct_result, parameters in test_cases:
            data_set = DataSet()
            data_set.data = data_frame

            result = data_set.get_data_attribute(**parameters)

            if type(correct_result) is pd.Series or type(correct_result) is pd.DataFrame:
                self.assertTrue(correct_result.equals(result))
            else:
                self.assertEqual(correct_result, result)

    def test_get_attribute_errors(self):
        test_cases = [
            (
                pd.DataFrame({"attr1": [1, 2, 3]}),
                KeyError,
                {"columns": "attr2"}
            ),
            (
                pd.DataFrame({"attr1": [1, 2, 3]}),
                KeyError,
                {"columns": "attr1", "row": 3}
            )
        ]

        for data_frame, correct_result, parameters in test_cases:
            data_set = DataSet()
            data_set.data = data_frame

            self.assertRaises(correct_result, data_set.get_data_attribute, **parameters)

    def test_set_attribute(self):
        data_set = DataSet()
        data_set.data = pd.DataFrame({"attr1": [1, 2, 3]})

        attributes_to_add = pd.DataFrame({"attr2": [4, 5, 6], "attr3": [7, 8, 9]})
        correct_result = pd.DataFrame({"attr1": [1, 2, 3], "attr2": [4, 5, 6], "attr3": [7, 8, 9]})

        data_set.set_data_attributes(attributes_to_add)

        self.assertTrue(correct_result.equals(data_set.data))

    def test_get_data_files(self):
        data_set = DataSet()
        data_set.data = pd.DataFrame({"lf_sensor": [1, 2, 3], "rf_sensor": [4, 5, 6], "p_sensor": [7, 8, 9]})
        correct_result = {"lf_sensor": 2, "rf_sensor": 5, "p_sensor": 8}

        result = data_set.get_data_files(1)

        for key in correct_result:
            self.assertEqual(correct_result[key], result[key])

    def test_get_file_names_first_format(self):
        data_set = DataSet()
        path.isfile = MagicMock()
        path.isfile.return_value = True

        data_set.data = pd.DataFrame({"File name root": ["root1", "root2", "root3"],
                                       "Day Code": ["dir1", "dir2", "dir3"],
                                       "LF Sensor No.": ["_lf_suffix1", "_lf_suffix2", "_lf_suffix3"],
                                       "RF Sensor No.": ["_rf_suffix1", "_rf_suffix2", "_rf_suffix3"],
                                       "P Sensor No.": ["_p_suffix1", "_p_suffix2", "_p_suffix3"]})

        correct_result = [{"lf_sensor": "dir1_Exported/root1_lf_suffix1", "rf_sensor": "dir1_Exported/root1_rf_suffix1", "p_sensor": "dir1_Exported/root1_p_suffix1"},
                          {"lf_sensor": "dir2_Exported/root2_lf_suffix2", "rf_sensor": "dir2_Exported/root2_rf_suffix2", "p_sensor": "dir2_Exported/root2_p_suffix2"},
                          {"lf_sensor": "dir3_Exported/root3_lf_suffix3", "rf_sensor": "dir3_Exported/root3_rf_suffix3", "p_sensor": "dir3_Exported/root3_p_suffix3"}]

        data_set.calc_data_files()

        for i in range(3):
            result = data_set.get_data_files(i)
            for key in correct_result[i]:
                self.assertEqual(data_set.data_dir + correct_result[i][key] + ".txt", result[key])

    def test_get_file_names_second_format(self):
        data_set = DataSet()
        path.isfile = MagicMock()
        path.isfile.return_value = False

        data_set.data = pd.DataFrame({"File name root": ["root1", "root2", "root3"],
                                       "Day Code": ["dir1", "dir2", "dir3"],
                                       "LF Sensor No.": ["_lf_suffix1", "_lf_suffix2", "_lf_suffix3"],
                                       "RF Sensor No.": ["_rf_suffix1", "_rf_suffix2", "_rf_suffix3"],
                                       "P Sensor No.": ["_p_suffix1", "_p_suffix2", "_p_suffix3"]})

        correct_result = [{"lf_sensor": "dir1_Exported/dir1_root1_lf_suffix1", "rf_sensor": "dir1_Exported/dir1_root1_rf_suffix1", "p_sensor": "dir1_Exported/dir1_root1_p_suffix1"},
                          {"lf_sensor": "dir2_Exported/dir2_root2_lf_suffix2", "rf_sensor": "dir2_Exported/dir2_root2_rf_suffix2", "p_sensor": "dir2_Exported/dir2_root2_p_suffix2"},
                          {"lf_sensor": "dir3_Exported/dir3_root3_lf_suffix3", "rf_sensor": "dir3_Exported/dir3_root3_rf_suffix3", "p_sensor": "dir3_Exported/dir3_root3_p_suffix3"}]

        data_set.calc_data_files()

        for i in range(3):
            result = data_set.get_data_files(i)
            for key in correct_result[i]:
                self.assertEqual(data_set.data_dir + correct_result[i][key] + ".txt", result[key])

if __name__ == '__main__':
    unittest.main()