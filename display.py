import matplotlib.pyplot as plt
import numpy as np
import re


def plot_learning_curves(train_score, test_score, num_points, clf_name='', save_fig=0):
    plt.figure()
    plt.title(clf_name + " Learning Curve")
    plt.xlabel("Number of points")
    increment_size = num_points / len(train_score)
    xs = [int(i * increment_size) for i in range(1, len(test_score))]
    xs.append(num_points)
    plt.plot(xs, train_score, label="Training Score")
    plt.plot(xs, test_score, label="Cross-Validation Score")
    plt.legend()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig("fig/" + clf_name + " Learning Curve " + str(save_fig), dpi=200)
        plt.close()

def plot_feature_importances(feature_names, feature_importances, var=None, sort=True, feature_selector=None,
                             clf_name='', save_fig=0):
    if sort:
        ordered_indices = np.argsort(feature_importances)[::-1]
    else:
        ordered_indices = range(len(feature_importances))

    if feature_selector is None:
        indices = ordered_indices
    else:
        indices = []
        for index in ordered_indices:
            if re.match(feature_selector, feature_names[index]) is not None:
                indices.append(index)

    plt.figure()
    plt.title(clf_name + "  Feature Importances")

    if var is None:
        plt.bar(range(len(indices)), feature_importances[indices], tick_label=[feature_names[i] for i in indices])
    else:
        std = np.sqrt(var)
        plt.bar(range(len(indices)), feature_importances[indices], yerr=std[indices], tick_label=[feature_names[i] for i in indices])

    plt.xticks(rotation=90)
    plt.tick_params(labelsize=5)
    plt.tight_layout()

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig("fig/" + clf_name + " Feature Importances" + str(save_fig), dpi=200)
        plt.close()