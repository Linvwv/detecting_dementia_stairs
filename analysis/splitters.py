"""
A file containing the implementations for various split strategies for cross validation including a random splitter and
a leave one out splitter.
"""

from sklearn.model_selection import train_test_split

class Splitter():
    """
    The abstract splitter class containing the interface that othe splitters should implement
    """
    def get_num_splits(self):
        """
        :return: This method should return the number of splits that a splitter can give. The split strategy therefore
                 specifies how many splits to run cross validation for.
        """
        pass

    def get_next_split(self, dataset):
        """
        Gets the next split of the dataset
        :param dataset: The dataset that needs to be split
        :return: train_dataset, cv_dataset
        """
        pass

    def reset(self):
        """
        Reset the splitter, this is equivalent to reinitialising it
        """
        pass

class LeaveOneOutSplitter(Splitter):
    """
    This class implements the leave one out split strategy. ie it splits the dataset into a cv dataset with data from a
    single participant and train dataset with the data from the other participants.
    """
    def __init__(self, split_param_name, split_param_values):
        self.split_param_name = split_param_name
        self.split_param_values = split_param_values
        self.split_counter = 0

    def get_num_splits(self):
        """
        :return: This method should return the number of splits that a splitter can give. The split strategy therefore
                 specifies how many splits to run cross validation for. This is the number of split values that exist -
                 the number of participants.
        """
        return len(self.split_param_values)

    def get_next_split(self, dataset):
        """
        Gets the next split of the dataset. The cv_dataset is data from one participant, the train dataset is data from
        the rest.
        :param dataset: The dataset that needs to be split
        :return: train_dataset, cv_dataset
        """
        # print("Split", self.split_counter + 1, "/", len(self.split_param_values))
        if self.split_counter == self.get_num_splits():
            return None, None

        if self.split_param_values[self.split_counter] not in set(dataset[self.split_param_name]):
            raise ValueError("The value on which the dataset is split must be in the dataset")

        train = dataset[dataset[self.split_param_name] != self.split_param_values[self.split_counter]]
        test = dataset[dataset[self.split_param_name] == self.split_param_values[self.split_counter]]

        self.split_counter += 1

        return train, test

    def reset(self):
        """
        Reset the splitter, this is equivalent to reinitialising it. Restarts splitter from the first split made
        """
        self.split_counter = 0


class RandomSplitter(Splitter):
    """
    Split the dataset randomly with the proportions for the cv size being specified by the user. This class implements
    the Random Permutation Cross Validation strategy and is not used anymore but is present for comparision purposes.
    """
    def __init__(self, num_splits=20, test_size=0.2):
        self.num_splits = num_splits
        self.test_size = test_size
        self.split_counter = 0

    def get_num_splits(self):
        """
        :return: This method should return the number of splits that a splitter can give. The split strategy therefore
                 specifies how many splits to run cross validation for. Specified by the user
        """
        return self.num_splits

    def get_next_split(self, dataset):
        """
        Gets the next split of the dataset. This split is done randomly with cv proportion being specified by the user.
        :param dataset: The dataset that needs to be split
        :return: train_dataset, cv_dataset
        """
        if self.split_counter == self.get_num_splits():
            return None, None

        self.split_counter += 1

        return train_test_split(dataset, test_size=self.test_size)

    def reset(self):
        """
        Reset the splitter, this is equivalent to reinitialising it
        """
        self.split_counter = 0