import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.utils

from preprocessing import preprocess

from analysis.model import Model


class ClassicalModel(Model):
    """
    A class encapsulating sklearn models. As such it is a wrapper around sklearns classifier classes.
    """
    def __init__(self, num_classes, clf_type, clf_params=None, label_mapper=None, upsampling=False):
        super().__init__(num_classes, clf_params, label_mapper, upsampling)
        self.clf_type = clf_type
        self.clf = None

    def spawn_clf(self, params=None):
        """
            Creates a classifier with the given parameters, or self.clf_parameters if params=None
            :param params: (optional) parameters for the new classifier
        """
        if params is None:
            params = self.clf_params

        # in case self.clf_params is also None
        if params is None:
            self.clf = self.clf_type()
        else:
            self.clf = self.clf_type(**params)

    def train(self, X, y, params=None):
        """
        Trains a new classifier with parameters specified.
        :param X: The features for the train dataset
        :param y: The labels for the train dataset.
        :param params: The parameters for the model to be trained. If None the the clf_parameters set in the constructor
                       should be used
        """
        self.spawn_clf(params=params)
        y = self.convert_labels(y)

        if self.upsampling:
            X, y = preprocess.up_sample(X, y)
        self.clf.fit(X, y)

    def predict(self, X):
        """
       Predict the labels of the a dataset
       :param X: The features to be predicted
       :return: The predicted labels for the data passed
       """
        return self.clf.predict(X)

    def score(self, X, y):
        """
        Score the current classifier on a test data set. The score used is the accuracy.
        :param X: A (n, m) numpy array, where n is the number of points in the dataset and m is the number of
            features
        :param y: A (n, ) numpy array, where n is the number of points in the dataset
        :return score: the accuracy of the classifier on the data set
        """
        y = self.convert_labels(y)

        return self.clf.score(X, y)

    def incremental_score(self, X_train, y_train, X_test, y_test, increments=30):
        """
       The scores during training. This is used to create the learning curves. In this case increments are implemented
       by training a classifiers on the train dataset incrementally and getting the scores on the entire train and test
       set
       :param X_train: The features of the train dataset
       :param y_train: The labels of the train dataset
       :param X_test: The features of the test dataset
       :param y_test: The features of the test dataset
       :return: (train_scores, test_scores) - each are lists of the scores.
       """
        train_scores = []
        test_scores = []

        increment_size = len(y_train) / increments
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        for i in range(1, increments + 1):
            end_index = int(increment_size * i) if i < increment_size else None
            self.train(X_train[:end_index], y_train[:end_index])
            train_scores.append(self.score(X_train, y_train))
            test_scores.append(self.score(X_test, y_test))

        return train_scores, test_scores

    def get_confusion_matrix(self, y_test, X_test=None, y_pred=None):
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
                y_pred = self.predict(X_test)
        y_test = self.convert_labels(y_test)

        return confusion_matrix(y_test, y_pred, labels=range(self.num_classes))

    def get_train_test_prediction(self, X_train, y_train, X_test, y_test):
        """
        A wrapper around predict that gets predictions on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_predictions, test_predictions) - each are lists of the predictions.
        """
        return list(zip(self.predict(X_train), self.convert_labels(y_train))), list(zip(self.predict(X_test), self.convert_labels(y_test)))

    def get_train_test_score(self, X_train, y_train, X_test, y_test):
        """
        Score the current classifier on a train and test data set
        :param X_train: A (n1, m) numpy array, where n is the number of points in the dataset and m is the number of
            features
        :param y_train: A (n1, ) numpy array, where n is the number of points in the dataset
        :param X_test: A (n2, m) numpy array, where n is the number of points in the dataset and m is the number of
            features
        :param y_test: A (n2, ) numpy array, where n is the number of points in the dataset
        :return train_score, test_score: he accuracy of the classifier on the two data sets
        """
        return self.score(X_train, y_train), self.score(X_test, y_test)

    def get_train_test_confusion_matrix(self, X_train, y_train, X_test, y_test):
        """
        A wrapper around get_confusion_matrix that gets confusion matrices on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_confusion_matrix, test_confusion_matrix) - each are numpy arrays of the confusion matrices
        """
        return self.get_confusion_matrix(y_train, X_test=X_train), self.get_confusion_matrix(y_test, X_test=X_test)

    def get_feature_importances(self):
        """
        Gets the feature importance of the classifier as define by sklearn's classifiers.
        :return: Feature importances, a numpy array of shape (number of features, )
        """
        return self.clf.feature_importances_








