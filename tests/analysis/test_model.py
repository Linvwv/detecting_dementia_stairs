import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np
import pandas as pd

import unittest

from analysis.model import Model

class TestModel(unittest.TestCase):
    def test_convert_labels(self):
        test_cases = [
            (
                # label mapper
                lambda y_in: y_in > float("-inf"),
                # input labels
                np.array([1, 2, 3, 4, 5]),
                # output labels
                np.array([True, True, True, True, True])
            ),
            (
                # label mapper
                lambda y_in: y_in > float("-inf"),
                # input labels
                pd.Series([1, 2, 3, 4, 5]),
                # output labels
                pd.Series([True, True, True, True, True])
            ),
            (
                # label mapper
                None,
                # input labels
                np.array([1, 2, 3, 4, 5]),
                # output labels
                np.array([1, 2, 3, 4, 5])
            ),
            (
                # label mapper
                None,
                # input labels
                pd.Series([1, 2, 3, 4, 5]),
                # output labels
                pd.Series([1, 2, 3, 4, 5])
            )
        ]
        for label_mapper, label_in, label_out in test_cases:
            model = Model(2, label_mapper=label_mapper)
            test_label_out = model.convert_labels(label_in)
            if type(label_out) is pd.Series:
                self.assertTrue(label_out.equals(test_label_out))
            else:
                self.assertTrue(np.array_equal(label_out, test_label_out))

if __name__ == '__main__':
    unittest.main()