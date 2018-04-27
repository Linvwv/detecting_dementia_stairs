import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import pandas as pd
from pandas.testing import assert_frame_equal

import unittest

from analysis.splitters import LeaveOneOutSplitter, RandomSplitter

class TestSplitters(unittest.TestCase):
    def test_random_splitter(self):
        test_cases = [
            (
                {"num_splits": 5, "test_size": 0.2},
                pd.DataFrame({"a": list(range(10)), "b": list(range(10))}),
            ),
            (
                {"num_splits": 5, "test_size": 0},
                pd.DataFrame({"a": list(range(10)), "b": list(range(10))}),
            ),
            (
                {"num_splits": 5, "test_size": 0.2},
                pd.DataFrame({"a": list(range(0)), "b": list(range(0))}),
            ),
            (
                {"num_splits": 0, "test_size": 0.2},
                pd.DataFrame({"a": list(range(5)), "b": list(range(5))}),
            )
        ]
        for splitter_params, dataset in test_cases:

            splitter = RandomSplitter(**splitter_params)
            self.assertEqual(splitter.get_num_splits(), splitter_params["num_splits"])
            for i in range(splitter_params["num_splits"]):
                train, test = splitter.get_next_split(dataset)
                self.assertEqual(train.shape, ((1 - splitter_params["test_size"]) * dataset.shape[0], dataset.shape[1]))
                self.assertEqual(test.shape, (splitter_params["test_size"] * dataset.shape[0], dataset.shape[1]))
                recombined_dataset = pd.concat([train, test])
                assert_frame_equal(dataset, recombined_dataset.sort_index(), check_names=False)

            train, test = splitter.get_next_split(dataset)
            self.assertIsNone(train)
            self.assertIsNone(test)

    def test_leave_one_out_splitter(self):
        test_cases = [
            pd.DataFrame({"a": list(range(5)), "b": list(range(5)), "c": [0, 0, 1, 1, 2]}),
            pd.DataFrame({"a": list(range(5)), "b": list(range(5)), "c": [1, 1, 1, 1, 1]})
        ]
        split_attr = "c"
        for dataset in test_cases:
            split_values = dataset["c"].unique()
            split_sizes = dataset["c"].value_counts()
            splitter = LeaveOneOutSplitter(split_attr, split_values)
            self.assertEqual(splitter.get_num_splits(), len(split_values))
            for i in range(splitter.get_num_splits()):
                train, test = splitter.get_next_split(dataset)
                self.assertEqual(train.shape, (dataset.shape[0] - split_sizes[split_values[i]], dataset.shape[1]))
                self.assertEqual(test.shape, (split_sizes[split_values[i]], dataset.shape[1]))
                recombined_dataset = pd.concat([train, test])
                assert_frame_equal(dataset, recombined_dataset.sort_index(), check_names=False)

            train, test = splitter.get_next_split(dataset)
            self.assertIsNone(train)
            self.assertIsNone(test)

    def test_leave_one_out_splitter_errors(self):
        split_attr = "c"
        split_values = [5, 9, 1, 2]

        dataset = pd.DataFrame({"a": list(range(5)), "b": list(range(5)), "c": [0, 0, 1, 1, 2]})
        splitter = LeaveOneOutSplitter(split_attr, split_values)
        self.assertRaises(ValueError, splitter.get_next_split, dataset)

        dataset = pd.DataFrame({"a": list(range(5)), "b": list(range(5))})
        splitter = LeaveOneOutSplitter(split_attr, split_values)
        self.assertRaises(KeyError, splitter.get_next_split, dataset)

if __name__ == '__main__':
    unittest.main()
