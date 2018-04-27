import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

import numpy as np
import tensorflow as tf

import unittest
from unittest.mock import MagicMock

from analysis.rnn_model import RNNModel
from preprocessing import preprocess

i = -1

class TestRNNModel(unittest.TestCase):
    # rnn_model.RNNModel.spawn_clf creates the network architecture. This was tested using visualisation. See
    # visualise_architecture.py in the same folder as this file. In the following tests this is mocked out.

    def test_fill_default_params(self):
        #check filling params with all defaults
        correct_default_params = {
            'rnn_cell_type': tf.contrib.rnn.GRUCell,
            'learning_rate': 0.001,
            'fully_connected_cells': 8,
            'num_stacks': 2,
            'num_rnn_units': 128
        }

        model = RNNModel(3, 8)
        self.assertDictEqual(correct_default_params, model.clf_params)

        # check some params filled in
        clf_params = {'num_stacks': 8, 'rnn_cell_type': tf.contrib.rnn.LSTMCell}
        model = RNNModel(3, 8, clf_params=clf_params)
        correct_params = {
            'rnn_cell_type': tf.contrib.rnn.LSTMCell,
            'learning_rate': 0.001,
            'fully_connected_cells': 8,
            'num_stacks': 8,
            'num_rnn_units': 128
        }
        self.assertDictEqual(correct_params, model.clf_params)

        # check second level dict param filling
        clf_params = {'num_stacks': 8, "random_param": 789}
        model = RNNModel(3, 8, clf_params=clf_params)
        correct_params = {
            'rnn_cell_type': tf.contrib.rnn.GRUCell,
            'learning_rate': 0.001,
            'random_param': 789,
            'fully_connected_cells': 8,
            'num_stacks': 8,
            'num_rnn_units': 128
        }
        self.assertDictEqual(correct_params, model.clf_params)

    def test_score(self):
        test_cases = [
            (
                # True values for y
                np.array([1, 0, 0, 0, 1]),
                # Predicted values for y (used as return value for mocked prediction method - tested separately)
                np.array([0, 1, 0, 0, 1]),
                # score
                0.6
            ),
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                3 / 9
            ),
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([1, 1, 1, 2, 2, 2, 0, 0, 0]),
                0
            ),
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                1
            ),
        ]
        for y_true, y_pred, score in test_cases:
            model = RNNModel(3, 8)
            model.predict = MagicMock()
            model.predict.return_value = y_pred

            # no params except y_true are used so just input dummy params
            self.assertAlmostEqual(model.score([], y_true, sequence_lengths=[]), score)

    def test_get_confusion_matrix(self):
        test_cases = [
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                np.array([[1] * 3] * 3)
            ),
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
            ),
            (
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                np.array([1, 1, 1, 2, 2, 2, 0, 0, 0]),
                np.array([[0, 3, 0], [0, 0, 3], [3, 0, 0]])
            )
        ]
        for y_true, y_pred, conf_mat in test_cases:
            model = RNNModel(3, 8)
            self.assertTrue(np.array_equal(model.get_confusion_matrix(y_true, y_pred=y_pred), conf_mat))

    def test_get_incremental_score(self):
        test_cases = [
            # scores to be returned in order by mock of model.score
            np.array([0, 0, 0.5, 0.3, 0.8, 0.6, 0.9, 0.7, 1, 0.8, 1, 0.9, 1, 1]),
            np.array([0, 144, 222, -0.29, 1, 1])

        ]
        for scores in test_cases:
            model = RNNModel(3, 8, num_epochs=int(len(scores) / 2))
            model.score = MagicMock()
            model.score.side_effect = scores
            model.spawn_clf = MagicMock()

            # mock tensorflow network attributes
            model.sess = MagicMock()
            model.sess.run = MagicMock()
            model.opt = MagicMock()
            model.x = MagicMock()
            model.y = MagicMock()
            model.sequence_lengths = MagicMock()

            preprocess.up_sample = MagicMock()

            # none of the inputs are actually used due to mocking so just give dummy inputs
            train_scores, test_scores = model.incremental_score([], [], [], [], train_sequence_lengths=[], test_sequence_lengths=[])
            self.assertListEqual(train_scores, list(scores[[i for i in range(0, len(scores), 2)]]))
            self.assertListEqual(test_scores, list(scores[[i for i in range(1, len(scores), 2)]]))

    def check_no_sequence_lengths_all_methods(self):
        rnn_model = RNNModel(2, 3)
        X_train = [[1,2,3], [4,5,6]]
        y_train = [0, 1]

        self.assertRaises(ValueError, rnn_model.train, X_train, y_train)
        self.assertRaises(ValueError, rnn_model.predict, X_train)
        self.assertRaises(ValueError, rnn_model.score, X_train, y_train)
        self.assertRaises(ValueError, rnn_model.incremental_score, X_train, y_train, X_train, y_train)
        self.assertRaises(ValueError, rnn_model.incremental_score, X_train, y_train, X_train, y_train, train_sequence_lengths=[])
        self.assertRaises(ValueError, rnn_model.incremental_score, X_train, y_train, X_train, y_train,
                          test_sequence_lengths=[])
        self.assertRaises(ValueError, rnn_model.get_confusion_matrix, X_train, y_train)

    def test_predict(self):
        test_cases = [
            (
                # Feature Matrix X
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                # the predictions to check for
                np.array([0, 1, 1]),
                # the RNN Model to use
                RNNModel(2, 3)
            ),
            (
                # Feature Matrix X
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                # the predictions to check for
                np.array([0, 1, 2]),
                # the RNN Model to use
                RNNModel(3, 3)
            )
        ]

        for X, true_predictions, rnn_model in test_cases:
            # mock rnn attributes that would be set by spawn_clf
            rnn_model.sess = MagicMock()
            rnn_model.prediction = MagicMock()
            rnn_model.x = MagicMock()
            rnn_model.sequence_lengths = MagicMock()
            rnn_model.sess.run.return_value = true_predictions
            self.assertTrue(np.array_equal(rnn_model.predict(X, []), true_predictions))

    def test_train(self):
        global i
        def side_effect(*args, **kwargs):
            global i
            if kwargs == {}:
                effect_collector.append('init')
            else:
                i += 1
                effect_collector.append(effects[i])

        test_cases = [
            (
                # RNN Model to be used
                RNNModel(2, 3),
                # X train to be used
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                # y train to be used
                np.array([0, 1, 1]),
                # sequence lengths
                np.array([5, 7, 2])
            ),
            (
                # RNN Model to be used
                RNNModel(2, 3, num_epochs=10),
                # X train to be used
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                # y train to be used
                np.array([0, 1, 1]),
                # sequence lengths
                np.array([5, 7, 2])
            ),
            (
                # RNN Model to be used
                RNNModel(3, 3, num_epochs=10),
                # X train to be used
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                # y train to be used
                np.array([0, 1, 2]),
                # sequence lengths
                np.array([5, 7, 2])
            )

        ]

        for rnn_model, X_train, y_train, sequence_lengths in test_cases:
            i = -1
            effect_collector = []
            effects = ['train' + str(i) for i in range(rnn_model.num_epochs)]
            self.assertEqual(len(effects), rnn_model.num_epochs)

            preprocess.upsample = MagicMock()
            rnn_model.spawn_clf = MagicMock()
            tf.global_variables_initializer = MagicMock()
            rnn_model.sess = MagicMock()
            rnn_model.sess.run.side_effect = side_effect
            rnn_model.opt = MagicMock()
            rnn_model.x = MagicMock()
            rnn_model.sequence_lengths = MagicMock()
            rnn_model.y = MagicMock()

            rnn_model.train(X_train, y_train, sequence_lengths)

            self.assertListEqual(effect_collector, ["init"] + effects)


if __name__ == '__main__':
    unittest.main()