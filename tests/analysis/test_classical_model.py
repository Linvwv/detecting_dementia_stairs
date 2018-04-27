import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.utils

import unittest
from unittest.mock import MagicMock

from analysis.classical_model import ClassicalModel

i = -1

class TestModel(unittest.TestCase):
    def test_spawn_clf(self):
        model = ClassicalModel(2, GradientBoostingClassifier, clf_params={"n_estimators": 0})

        self.assertEqual(GradientBoostingClassifier, model.clf_type)
        self.assertIsNone(model.clf)

        #create clf and check that it is created correctly
        model.spawn_clf()
        self.assertEqual(GradientBoostingClassifier, model.clf_type)
        self.assertIsNotNone(model.clf)
        self.assertEqual(type(model.clf), model.clf_type)
        self.assertEqual(model.clf.n_estimators, 0)

        # spawn a new clf and check that a new one is created and created correctly
        old_clf = model.clf
        model.spawn_clf({"n_estimators": 1})
        self.assertEqual(GradientBoostingClassifier, model.clf_type)
        self.assertIsNotNone(model.clf)
        self.assertEqual(type(model.clf), model.clf_type)
        self.assertNotEqual(old_clf, model.clf)
        self.assertEqual(model.clf.n_estimators, 1)

    def test_train_and_predict(self):
        x = pd.DataFrame([[-1], [-5], [1], [5]])
        y = pd.Series([0, 0, 1, 1])
        model = ClassicalModel(2, GradientBoostingClassifier, clf_params={"n_estimators": 1, "max_depth": 1})

        model.train(x, y)
        y_pred = pd.Series(model.predict(x))

        self.assertTrue(y.equals(y_pred))

    def test_score(self):
        x = pd.DataFrame([[-1], [-5], [1], [5]])
        y = pd.Series([0, 0, 1, 2])
        model = ClassicalModel(2, GradientBoostingClassifier, clf_params={"n_estimators": 1, "max_depth": 1}, label_mapper=lambda x: x > 0)

        model.train(x, y)
        score = model.score(x, y)
        y_pred = pd.Series(model.predict(x))
        self.assertTrue(y_pred.equals(pd.Series([False, False, True, True])))

        self.assertEqual(score, 1)

    def test_incremental_score(self):
        X_train = [[-1], [-2], [3], [4]]
        y_train = [0, 1, 2, 2]
        X_test = [[5], [6]]
        y_test = [2, 2]
        model = ClassicalModel(2, GradientBoostingClassifier, clf_params={"n_estimators": 1, "max_depth": 2})
        sklearn.utils.shuffle = MagicMock()
        sklearn.utils.shuffle.return_value = X_train, y_train

        train_scores, test_scores = model.incremental_score(X_train, y_train, X_test, y_test, increments=2)

        # will get 0.5 of training dataset right first as it only sees first 2 points, then 1 as it will see all points
        # and can fully capture all the data
        self.assertListEqual(train_scores, [0.5, 1.0])
        # will get 0 of testing dataset right first as it does not see points with label 2, then 1 as it will see all
        # points and can fully capture all the data
        self.assertListEqual(test_scores, [0., 1.0])

    def test_get_confusion_matrix(self):
        model = ClassicalModel(3, GradientBoostingClassifier, clf_params={"n_estimators": 1, "max_depth": 2})
        model.predict = MagicMock()
        model.predict.return_value = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]

        conf_mat = model.get_confusion_matrix(y_true, X_test=[[1], [2], [3]])

        true_conf_mat = np.array([[1,1,1],[1,1,1],[1,1,1]])

        self.assertTrue(np.array_equal(conf_mat, true_conf_mat))

if __name__ == '__main__':
    unittest.main()
