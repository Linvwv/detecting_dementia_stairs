
class Model():
    """
    An abstract parent class for the model that defines a common interface for all models implemented in this project.
    Features are denoted as X and have shape of (num_data_points, num_features), y denotes labels and have a shape of
    (num_data_points,)
    """
    def __init__(self, num_classes, clf_params=None, label_mapper=None, upsampling=False, *args, **kwargs):
        """
        Constructor for the model.
        :param num_classes: The number of classes that participants should be classified into
        :param clf_params: The hyper-parameters for the model
        :param label_mapper: A function that maps labels to the required number of labels. Or None if this is not needed.
        :param upsampling: A boolean flag that allows the model to conduct upsampling.
        :param args: Other arguments that the subclasses may use
        :param kwargs: Other arguments that the subclasses may use
        """
        self.num_classes = num_classes
        self.label_mapper = label_mapper
        self.clf_params = clf_params
        self.upsampling = upsampling

    def convert_labels(self, y):
        """
        Converts the labels into the required number of labels as specified by the label mapper function
        :param y: The labels to be converted
        :return: The converted labels - the return values of the label_mapper
        """
        if self.label_mapper is None:
            return y
        else:
            return self.label_mapper(y)

    def spawn_clf(self):
        """
        Spawns a fresh classifier
        """
        pass

    def train(self, X, y, params=None):
        """
        Trains a classifier with parameters specified.
        :param X: The features for the train dataset
        :param y: The labels for the train dataset.
        :param params: The parameters for the model to be trained. If None the the clf_parameters set in the constructor
                       should be used
        """
        pass

    def predict(self, X):
        """
        Predict the labels of the a dataset
        :param X: The features to be predicted
        :return: The predicted labels for the data passed
        """
        pass

    def score(self, X, y):
        """
        Score the model.
        :param X: The features to be inputted into the model
        :param y: The true labels against which to score the model
        :return: The score of the model.
        """
        pass

    def incremental_score(self, X_train, y_train, X_test, y_test):
        """
        The scores during training. This is used to create the learning curves.
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_scores, test_scores) - each are lists of the scores.
        """
        pass

    def get_confusion_matrix(self, y, X=None, y_pred=None):
        """
        Calculates the confusion matrix on some dataset. This can either be done by passing features to this method and
        calling the predict method to get y_pred or by using some precomputed y_pred. If y_pred is not None then the
        second method is used
        :param y: The true labels
        :param X: The features of the dataset, or None
        :param y_pred: The predictions made by the model or None.
        :return: The confusion matrix. A numpy array of shape (num_classes, num_classes).
        """
        pass

    def get_train_test_prediction(self, X_train, y_train, X_test, y_test):
        """
        A wrapper around predict that gets predictions on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_predictions, test_predictions) - each are lists of the predictions.
        """
        pass

    def get_train_test_confusion_matrix(self, X_train, y_train, X_test, y_test):
        """
        A wrapper around get_confusion_matrix that gets confusion matrices on both the train and test datasets
        :param X_train: The features of the train dataset
        :param y_train: The labels of the train dataset
        :param X_test: The features of the test dataset
        :param y_test: The features of the test dataset
        :return: (train_confusion_matrix, test_confusion_matrix) - each are numpy arrays of the confusion matrices
        """
        pass

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
        pass