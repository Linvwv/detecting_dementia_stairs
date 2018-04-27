import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import pandas as pd
import numpy as np

import unittest
from unittest.mock import MagicMock

from analysis import runners
from analysis.model import Model
from analysis.splitters import Splitter

i = -1

class TestRunners(unittest.TestCase):
    # test_get_score_all_splits
    def test_get_property_for_single_run(self):
        model = Model(3)
        # mock model. Model is tested separately
        model.train = MagicMock()
        model.get_train_test_score = MagicMock()
        model.get_train_test_score.return_value = [1, 0.9]
        splitter_func = lambda d: (d[:3], d[3:])
        # dummy dataset that is not used
        dataset = pd.DataFrame({'x': [1,2,3,4,5], 'y':[1,1,0,0,0]})

        result = runners.get_prop_single_split(dataset, ['x'], 'y', model, splitter_func)

        self.assertTupleEqual(result, (1, 0.9))

    def test_get_property_for_splits(self):
        global i
        i = -1
        props = [(1, 0.5), (2, 1), (3, 1.5)]
        def prop_func(train_X, train_y, test_X, test_y):
            global i
            i += 1
            return props[i]
        # mock model. Model is tested separately
        model = Model(3)
        model.train = MagicMock()
        # mock splitter. Splitter is tested separately
        splitter = Splitter()
        splitter.get_num_splits = MagicMock(return_value=3)
        splitter.get_next_split = MagicMock(return_value=(pd.DataFrame({'x': [1, 2, 3], 'y': [1, 1, 0]}), pd.DataFrame({'x': [4, 5], 'y': [0, 0]})))
        # dummy dataset that is not used
        dataset = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 1, 0, 0, 0]})

        train_prop, test_prop = runners.get_property_for_splits(dataset, ['x'], 'y', model, splitter, property_getter=prop_func)

        self.assertListEqual(train_prop, [1, 2, 3])
        self.assertListEqual(test_prop, [0.5, 1, 1.5])

    def test_get_avg_prop_all_splits(self):
        global i
        i = -1
        props = [(1, 0.5), (2, 1), (3, 1.5)]
        def prop_func(train_X, train_y, test_X, test_y):
            global i
            i += 1
            return props[i]
        # mock model. Model is tested separately
        model = Model(3)
        model.train = MagicMock()
        # mock splitter. Splitter is tested separately
        splitter = Splitter()
        splitter.get_num_splits = MagicMock(return_value=3)
        splitter.get_next_split = MagicMock(return_value=(pd.DataFrame({'x': [1, 2, 3], 'y': [1, 1, 0]}), pd.DataFrame({'x': [4, 5], 'y': [0, 0]})))
        # dummy dataset that is not used
        dataset = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 1, 0, 0, 0]})

        train_prop, test_prop = runners.get_avg_prop_all_splits(dataset, ['x'], 'y', model, splitter, property_getter=prop_func)

        self.assertEqual(train_prop, 2)
        self.assertEqual(test_prop, 1)

    def test_get_total_score_all_splits(self):
        runners.get_property_for_splits = MagicMock()
        runners.get_property_for_splits.return_value = (np.array([[(1, 1), (0, 0), (1, 1)], [(1, 1), (1, 1), (1, 1)], [(0, 0), (0, 0), (1, 1)]]),
                                                        np.array([[(1, 0), (0, 0), (1, 0)], [(1, 1), (1, 0), (1, 1)], [(0, 0), (0, 0), (1, 1)]]))
        model = MagicMock()
        splitter = MagicMock()
        dataset = MagicMock()

        train_acc, test_acc = runners.get_total_score_all_splits(dataset, ['x'], 'y', model, splitter)

        self.assertEqual(train_acc, 1)
        self.assertEqual(test_acc, 6 / 9)

    def test_hyperparam_tuning(self):
        model = MagicMock()
        splitter = MagicMock()
        dataset = MagicMock()

        runners.get_total_score_all_splits = MagicMock()
        # chooses based on max cv score so first item in tuple is ignored
        runners.get_total_score_all_splits.side_effect = [(0, 1), (0, 3), (0, 0), (0, -4)]

        try_params = {"n_estimators": [100, 200], "max_depth": [1, 2]}

        best_param, _ = runners.run_hyperparameter_tuning(dataset, ["x"], "y", model, splitter, try_params, num_repeats=1)

        self.assertEqual(best_param[1], 3)

    def test_hyperparam_tuning_with_repeats(self):
        model = MagicMock()
        splitter = MagicMock()
        dataset = MagicMock()

        runners.get_total_score_all_splits = MagicMock()
        # chooses based on max cv score so first item in tuple is ignored
        runners.get_total_score_all_splits.side_effect = [(0, 1), (0, 1.5),
                                                          (0, 3), (0, 3.25),
                                                          (0, 0), (0, 0),
                                                          (0, -4), (0, -3)]

        try_params = {"n_estimators": [100, 200], "max_depth": [1, 2]}

        best_param, param_scores = runners.run_hyperparameter_tuning(dataset, ["x"], "y", model, splitter, try_params, num_repeats=2)

        self.assertEqual(best_param[1], (3 + 3.25)/2)

if __name__ == '__main__':
    unittest.main()