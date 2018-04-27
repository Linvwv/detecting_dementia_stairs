import numpy as np

from analysis.classical_model import ClassicalModel
import itertools

import display
from analysis.rnn_model import RNNModel


def get_prop_single_split(dataset, features_names, label_name, model, splitter_func, property_getter=None, get_train_size=False, train=True):
    """
    Gets a property of the classifier specified by a property_getter for a single split that can be retrieved using a
    splitter_func on the dataset.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter_func: A splitter function that returns the split
    :param property_getter: A function that returns the desired property on both the train and test sets. Uses the
                            models get_train_test_score method if None
    :param get_train_size: A boolean flag that allows the size of the train dataset to be retrieved
    :param train: A flag that allows for training to be switched on and off
    :return: train_property, test_property - the property on the dataset split by the splitter_function
    """
    if property_getter is None:
        property_getter = model.get_train_test_score

    train_dataset, test_dataset = splitter_func(dataset)

    if type(model) is RNNModel:
        if train:
            model.train(train_dataset[features_names], train_dataset[label_name], train_dataset["sequence_lengths"])
        train_score, test_score = property_getter(train_dataset[features_names], train_dataset[label_name],
                                                  test_dataset[features_names], test_dataset[label_name],
                                                  train_dataset["sequence_lengths"], test_dataset["sequence_lengths"])
    else:
        if train:
            model.train(train_dataset[features_names], train_dataset[label_name])
        train_score, test_score = property_getter(train_dataset[features_names], train_dataset[label_name],
                                                    test_dataset[features_names], test_dataset[label_name])
    if get_train_size:
        return train_score, test_score, len(train_dataset[label_name])
    else:
        return train_score, test_score


def plot_learning_curve_single_split(dataset, feature_names, label_name, model, splitter_func, clf_name, save_fig=False):
    """
    Plots the learning curves for a single split of the dataset on the model specified
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter_func: A splitter function that returns the split
    :param clf_name: The name of the classifier to put in the title of the plot
    :param save_fig: A flag of whether or not to save the figue. False displays the figure, True saves the figure and
                     doesn't display it
    """
    train_scores, test_scores, train_size = get_prop_single_split(dataset, feature_names, label_name, model, splitter_func,
                                                                  model.incremental_score, get_train_size=True, train=False)
    display.plot_learning_curves(train_scores, test_scores, train_size, clf_name=clf_name,
                                 save_fig=1 if save_fig else 0)


def get_property_for_splits(dataset, feature_names, label_name, model, splitter, property_getter=None,
                            get_train_size=False, train=True):
    """
    Gets a property (specified by property_getter) across all the splits of dataset with given features and label. It
    does this using a splitting strategy which is one of the Splitters.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param property_getter: A function that returns the desired property on both the train and test sets. Uses the
                            models get_train_test_score method if None
    :param get_train_size: A boolean flag that allows the size of the train dataset to be retrieved
    :param train: A flag that allows for training to be switched on and off
    :return: train_property, test_property - the properties on all splits of the dataset split by the splitter
    """
    train_prop = []
    test_prop = []
    train_size = []

    for i in range(splitter.get_num_splits()):
        if get_train_size:
            curr_train_prop, curr_test_prop, curr_train_size = get_prop_single_split(dataset, feature_names, label_name,
                                                                                     model, splitter.get_next_split, property_getter,
                                                                                     get_train_size=get_train_size, train=train)
        else:
            curr_train_prop, curr_test_prop = get_prop_single_split(dataset, feature_names, label_name, model,
                                                                    splitter.get_next_split, property_getter,
                                                                    get_train_size=get_train_size, train=train)
        train_prop.append(curr_train_prop)
        test_prop.append(curr_test_prop)
        if get_train_size:
            train_size.append(curr_train_size)

    if get_train_size:
        return train_prop, test_prop, train_size
    else:
        return train_prop, test_prop


def get_total_score_all_splits(dataset, feature_names, label_name, model, splitter):
    """
    Gets the score of the train and test sets across all the splits. This is done by getting the predictions for each
    split and aggregating these into all the predictions and then calculating the the accuracy.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :return: train_score, test_score - the scores on the train and test dataset across all splits
    """
    predictions = get_property_for_splits(dataset, feature_names, label_name, model, splitter,
                                                                model.get_train_test_prediction)

    correct = [0, 0]
    total = [0, 0]

    for i in range(len(predictions)):
        for prediction in predictions[i]:
            correct[i] += sum(map(lambda x: x[0] == x[1], prediction))
            total[i] += len(prediction)

    return [correct[i] / total[i] for i in range(len(correct))]


def get_avg_prop_all_splits(dataset, feature_names, label_name, model, splitter, property_getter=None, train=True):
    """
    Gets the average of a property across all splits.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param property_getter: A function that returns the desired property on both the train and test sets. Uses the
                            models get_train_test_score method if None
    :param train: A flag that allows for training to be switched on and off
    :return: avg_train_prop, avg_test_prop - the average property across all split on the training and testing datasets
    """

    scores = get_property_for_splits(dataset, feature_names, label_name, model, splitter, property_getter, train=train)

    return [np.mean(dataset_score, axis=0) for dataset_score in scores]


def plot_avg_learning_curve_all_splits(dataset, feature_names, label_name, model, splitter, clf_name, save_fig=False):
    """
    Plots the average learning curves across all the splits.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param clf_name: The name of the classifier to put in the title of the plot
    :param save_fig: A flag of whether or not to save the figue. False displays the figure, True saves the figure and
                     doesn't display it
    """
    train_curve, test_curve = get_avg_prop_all_splits(dataset, feature_names, label_name, model, splitter,
                                                      property_getter=model.incremental_score, train=False)

    if type(model) is ClassicalModel:
        display.plot_learning_curves(train_curve, test_curve, 500, "Average " + clf_name, save_fig=1 if save_fig else 0)
    elif type(model) is RNNModel:
        display.plot_learning_curves(train_curve, test_curve, 20, "Average " + clf_name, save_fig=1 if save_fig else 0)


def plot_learning_curves_all_splits(dataset, feature_names, label_name, model, splitter, clf_name, save_fig=False):
    """
    Plots individual learning curves for each split.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param clf_name: The name of the classifier to put in the title of the plot
    :param save_fig: A flag of whether or not to save the figue. False displays the figure, True saves the figure and
                     doesn't display it
    """
    train_curves, test_curves, train_sizes = get_property_for_splits(dataset, feature_names, label_name, model,
                                                                        splitter, property_getter=model.incremental_score,
                                                                        get_train_size=True, train=False)

    for i in range(len(train_curves)):
        display.plot_learning_curves(train_curves[i], test_curves[i], train_sizes[i], clf_name,
                                     save_fig=1 if save_fig else 0)


def plot_feature_importances(dataset, feature_names, label_name, model, splitter, clf_name, show_var=True, sort=True,
                             feature_selector=None, save_fig=False):
    """
    Plots the average importance of each feature across all splits
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param clf_name: The name of the classifier to put in the title of the plot
    :param show_var: A flag that allows the variance of each feature importance to be displayed in the plot
    :param sort: A boolean flag that allows features to be sorted from most important to least important
    :param feature_selector: A regex that allows for selecting certain types of features and only displaying these
    :param save_fig: A flag of whether or not to save the figue. False displays the figure, True saves the figure and
                     doesn't display it
    """
    if type(model) is not ClassicalModel:
        raise TypeError("The model must be one of sklearns analysis with feature importances implemented")
    property_getter = lambda train_X, train_y, test_X, test_y: [[], model.get_feature_importances()]
    _, feature_importances = get_property_for_splits(dataset, feature_names, label_name, model, splitter, property_getter)

    avg_importances = np.mean(feature_importances, axis=0)
    if show_var:
        var_importances = np.var(feature_importances, axis=0)
    else:
        var_importances = None

    display.plot_feature_importances(feature_names, avg_importances, var=var_importances, sort=sort,
                                     feature_selector=feature_selector, clf_name=clf_name,
                                     save_fig=1 if save_fig else 0)

def run_hyperparameter_tuning(dataset, feature_names, label_name, model, splitter, params, threshold=None, num_repeats=10, is_list=False):
    """
    Conducts hyperparameter tuning given a set of parameter to try out returns the best parameter along with the scores
    for each parameter tried. Tries each parameter repeatedly getting the average score to prevent a spurious good
    result from winning.
    :param dataset: The dataset on which to run the analysis
    :param features_names: The feature names used in the current analysis
    :param label_name: The name of the label attribute to be used. The column in which the label resides in the dataset
    :param model: The model class to be used for doing the analysis
    :param splitter: The implementation of a particular splitter function. Should be one of the splitter classes
                     implemented
    :param params: The params to be tried if is list is true then it is a list of dictionaries. Each dictionary
                   represents one setting for the parameters. If is_list is false then this is a dictionary mapping from
                   the name of the params to a list of parameter values. The cartesian product is calculated on this
                   dictionary to get all combinations of the features in the dictionary.
    :param threshold: A threshold, if under this threshold for any of the multiple runs then do not run it again. If
                      None then no threshold is used
    :param num_repeats: The number of times to repeat running the model across all splits. The score is averaged across
                        these splits to prevent a spurious good result from winning
    :param is_list: A flag that allows parameters to be specified as in two different ways - as precomputed permutations
                    or sets of values from which to construct the permutations
    :return: Returns the best parameter based on the accuracy, also returns a list of tuples. Each tuple is a parameter
             settings and the score for that setting.
    """
    if params is None:
        param_permutations = [None]
    elif is_list:
        param_permutations = params
    else:
        param_permutations = [dict(zip(params.keys(), el)) for el in list(itertools.product(*params.values()))]

    scores = []
    for param in param_permutations:
        cum_score_for_param = 0
        if type(model) is RNNModel:
            model.clf_params = model.fill_default_params(param)
        else:
            model.clf_params = param
        for i in range(num_repeats):
            score = get_total_score_all_splits(dataset, feature_names, label_name, model, splitter)[1]
            splitter.reset()
            cum_score_for_param += score
            if threshold is not None and score < threshold:
                break
        scores.append(cum_score_for_param / (i + 1))

    max_index = np.argmax(scores)

    param_scores = list(zip(param_permutations, scores))

    return param_scores[max_index], param_scores
