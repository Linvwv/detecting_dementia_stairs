from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from preprocessing import preprocess
import settings
from analysis.model import Model


class RNNModel(Model):
    """
    A class encapsulating RNN models in TensorFlow. As such it is a wrapper around TensorFlow's functionality classes.
    """
    def __init__(self, num_classes, num_features, label_mapper=None, clf_params=None, num_epochs=20, upsampling=False):
        super().__init__(num_classes, clf_params, label_mapper, upsampling)
        self.num_classes = num_classes
        self.label_mapper = label_mapper
        self.clf_params = self.fill_default_params(clf_params)
        self.num_epochs = num_epochs
        self.upsampling = upsampling
        self.num_features = num_features

    def fill_default_params(self, params):
        """
        Fills params with default values if any params have not been set.
        :param params: The params that have been set.
        :return: The params with any missing params filled with their default values.
        """
        if params is None:
            params = {}

        if "rnn_cell_type" not in params:
            params["rnn_cell_type"] = tf.contrib.rnn.GRUCell
        if "num_stacks" not in params:
            params["num_stacks"] = 2
        if "fully_connected_cells" not in params:
            params["fully_connected_cells"] = 8
        if "num_rnn_units" not in params:
            params["num_rnn_units"] = 128
        if "learning_rate" not in params:
            params["learning_rate"] = 0.001

        return params

    def spawn_clf(self, params=None, num_windows=None):
        """
        Creates the tensorflow network architecture for the desired RNN. With the parameters specified and the number of
        windows specified
        :param params: The hyper-parameters to be used for the classifier spawned
        :param num_windows: The number of windows to be used for the classifier. settings contains a constant that has
                            the maximum number of frames in any trial.
        """
        # reset graph to default and create a session
        if num_windows is None:
            num_windows = settings.num_windows
        tf.reset_default_graph()
        self.sess = tf.Session()

        # setup params
        if params is None:
            params = self.clf_params
        else:
            params = self.fill_default_params(params)

        # The inputs to the RNN. These are taken in the same form as other classifiers and then are reshaped so that the
        # RNN takes all data from one timestep as a single input.
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.num_features], name="input")
            self.x_reshaped = tf.reshape(self.x, [-1, int(num_windows), int(self.num_features / num_windows)], name="reshape")

        # The labels for optimising the network during training. These are encoded as a one_hot array so that they can
        # be compared to the softmax outputs
        with tf.name_scope('labels'):
            self.y = tf.placeholder(tf.int64, [None], name="labels")
            y_one_hot = tf.one_hot(self.y, self.num_classes, name="one_hot")

        # The lengths of the sequences without any extra padded zeros that may exist and other information reuqired for
        # selecting only the last prediction in the time series
        with tf.name_scope('sequence_lengths'):
            self.sequence_lengths = tf.placeholder(tf.int32, [None])
            batch_size = tf.shape(self.x)[0]

        # creates the base RNN layer with the given type, parameters and number of stacks.
        stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell([params["rnn_cell_type"](num_units=params["num_rnn_units"]) for i in range(params["num_stacks"])])
        # The actual RNN
        self.rnn_outputs, _ = tf.nn.dynamic_rnn(stacked_rnn_cell, self.x_reshaped, dtype="float32", sequence_length=self.sequence_lengths)
        # Selects the last output of the RNN and only provides this to the rest of the network. Allowing the network to
        # accumulate all information across the trial but not learn from padded zeroes.
        with tf.name_scope('last_in_series'):
            self.rnn_prediction = tf.gather_nd(self.rnn_outputs, indices=tf.stack([tf.range(batch_size), self.sequence_lengths - 1], axis=1))

        # The MLP/Feedforward part of the network that allows for predictions to be made
        relu = tf.nn.relu(self.rnn_prediction, name="rnn_relu")
        fully_connected = tf.contrib.layers.fully_connected(relu, params["fully_connected_cells"])

        # The output layer
        output = tf.contrib.layers.fully_connected(fully_connected, self.num_classes, activation_fn=None)

        # Make predictions by finding the maximum output and assigning the label corresponding to this
        self.prediction = tf.argmax(output, 1, name="prediction")

        # Softmax cross entorpy loss is used to optimise the network
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_one_hot))

        # The optimisation step - a minimization step of the Adam Optimiser on the loss
        self.opt = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(loss)

    def train(self, X, y, sequence_lengths=None, params=None, spawn_new=True, get_incremental_score=False, test_X=None, test_y=None, test_sequence_lengths=None):
        """
        Trains a classifier with parameters specified.
        :param X: The features for the train dataset
        :param y: The labels for the train dataset.
        :param params: The parameters for the model to be trained. If None the the clf_parameters set in the constructor
                       should be used
        :param sequence_lengths: The lengths of the sequences used as inputs without any trailing zeros.
        :param spawn_new: A flag that allows spawning a new classifier to be turned off. Used in get_incremental_scores
        """
        if sequence_lengths is None:
            raise ValueError("sequence_lengths cannot be None")
        y = self.convert_labels(y)
        if get_incremental_score:
            train_scores = []
            test_scores = []

        if self.upsampling:
            X, y, sequence_lengths = preprocess.up_sample(X, y, sequence_lengths)

        if spawn_new:
            self.spawn_clf(params)
            init = tf.global_variables_initializer()
            self.sess.run(init)

        # run optimisation step for the number of epoch specified
        for i in range(self.num_epochs):
            self.sess.run(self.opt, feed_dict={self.x: X, self.y: y, self.sequence_lengths: sequence_lengths})
            if get_incremental_score:
                train_scores.append(self.score(X, y, sequence_lengths))
                test_scores.append(self.score(test_X, test_y, test_sequence_lengths))

        if get_incremental_score:
            return train_scores, test_scores

    def write_graph_for_visualisation(self):
        """
        Writes information needed for visualising the network to the log files.
        """
        writer = tf.summary.FileWriter("tmp/rnn_data")
        writer.add_graph(self.sess.graph)


    def predict(self, X, sequence_lengths=None):
        """
        Predict the labels of the a dataset
        :param X: The features to be predicted
        :param sequence_lengths:  The lengths of the sequences used as inputs without any trailing zeros.
        :return: The predicted labels for the data passed
        """
        if sequence_lengths is None:
            raise ValueError("sequence_lengths cannot be None")

        return self.sess.run(self.prediction, feed_dict={self.x: X, self.sequence_lengths: sequence_lengths})

    def score(self, X_test, y_test, sequence_lengths=None):
        """
        Score the model.
        :param X: The features to be inputted into the model
        :param y: The true labels against which to score the model
        :param sequence_lengths:  The lengths of the sequences used as inputs without any trailing zeros.
        :return: The score of the model.
        """
        if sequence_lengths is None:
            raise ValueError("sequence_lengths cannot be None")

        y_test = self.convert_labels(y_test)
        y_pred = self.predict(X_test, sequence_lengths=sequence_lengths)

        return np.mean(y_test == y_pred)

    def incremental_score(self, X_train, y_train, X_test, y_test, train_sequence_lengths=None, test_sequence_lengths=None):
        """
        The scores during training. This is used to create the learning curves. The increments used are the number of
        epochs. So a single epoch of training is done then the model is scored and this is repeated until the desired
        number of epochs is completed.
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :param train_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                       trailing zeros.
        :param test_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                      trailing zeros.
        :return: (train_scores, test_scores) - each are lists of the scores.
        """
        if train_sequence_lengths is None or train_sequence_lengths is None:
            raise ValueError("sequence_lengths cannot be None")

        return self.train(X_train, y_train, sequence_lengths=train_sequence_lengths, get_incremental_score=True, test_X=X_test, test_y=y_test, test_sequence_lengths=test_sequence_lengths)

    def get_confusion_matrix(self, y_test, X_test=None, y_pred=None, sequence_lengths=None):
        """
        Calculates the confusion matrix on some dataset. This can either be done by passing features to this method and
        calling the predict method to get y_pred or by using some precomputed y_pred. If y_pred is not None then the
        second method is used
        :param y: The true labels
        :param X: The features of the dataset, or None
        :param y_pred: The predictions made by the model or None.
        :return: The confusion matrix. A numpy array of shape (num_classes, num_classes).
        """
        if y_pred is None:
            if X_test is None:
                return None
            else:
                if sequence_lengths is None:
                    raise ValueError("sequence_lengths cannot be None")
                y_pred = self.predict(X_test, sequence_lengths=sequence_lengths)

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        y_test = list(self.convert_labels(y_test))

        for true_group in range(self.num_classes):
            # get the predictions for that true group

            preds_for_group = y_pred[list(filter(lambda i: y_test[i] == true_group, range(len(y_test))))]

            # get the counts for each predicted group
            prediction_counts = Counter(preds_for_group)
            for predicted_group in prediction_counts:
                confusion_matrix[true_group][predicted_group] = prediction_counts[predicted_group]

        return confusion_matrix

    def get_train_test_prediction(self, X_train, y_train, X_test, y_test, train_sequence_lengths=None, test_sequence_lengths=None):
        """
        A wrapper around predict that gets predictions on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :param train_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                       trailing zeros.
        :param test_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                      trailing zeros.
        :return: (train_predictions, test_predictions) - each are lists of the predictions.
        """
        return list(zip(self.predict(X_train, sequence_lengths=train_sequence_lengths), self.convert_labels(y_train))), list(zip(self.predict(X_test, sequence_lengths=test_sequence_lengths), self.convert_labels(y_test)))

    def get_train_test_score(self, X_train, y_train, X_test, y_test, train_sequence_lengths=None, test_sequence_lengths=None):
        """
        Score the current classifier on a train and test data set
        :param X_train: A (n1, m) numpy array, where n is the number of points in the dataset and m is the number of
            features
        :param y_train: A (n1, ) numpy array, where n is the number of points in the dataset
        :param X_test: A (n2, m) numpy array, where n is the number of points in the dataset and m is the number of
            features
        :param y_test: A (n2, ) numpy array, where n is the number of points in the dataset
        :param train_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                       trailing zeros.
        :param test_sequence_lengths: The lengths of the sequences used as inputs in the train dataset without any
                                      trailing zeros.
        :return train_score, test_score: he accuracy of the classifier on the two data sets
        """
        return self.score(X_train, y_train, sequence_lengths=train_sequence_lengths), self.score(X_test, y_test, sequence_lengths=test_sequence_lengths)

    def get_train_test_confusion_matrix(self, X_train, y_train, X_test, y_test, train_sequence_lengths=None, test_sequence_lengths=None):
        """
        A wrapper around get_confusion_matrix that gets confusion matrices on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_confusion_matrix, test_confusion_matrix) - each are numpy arrays of the confusion matrices
        """
        return self.get_confusion_matrix(y_train, X_test=X_train, sequence_lengths=train_sequence_lengths), \
               self.get_confusion_matrix(y_test, X_test=X_test, sequence_lengths=test_sequence_lengths)