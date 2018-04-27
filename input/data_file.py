import numpy as np
import pandas as pd

class DataFile():
    """
    A class encapsulating the data files. One of such object encapsulates one data file produced by a single sensor
    during a single trial or traversal of the stairs. Therefore, three such objects encapsulate the data for a trial
    (one for each sensor - left foot, right foot, pelvis). Each data file is encapsulated as a pandas dataframe with
    methods additional functionality for retrieving multidimensional data attribute in the way in which they appear in
    the data files.
    """
    def __init__(self, file_path, header_rows=4):
        """
        The constructor for the DataFile object.
        :param file_path: The path to the data file
        :param header_rows: the number of header rows/lines. These are the rows before the tabular data starts.
        """
        self.file_path = file_path
        self.header_rows=header_rows

    def get_frequency(self, name_in_file="Update Rate: "):
        """
        Gets the frequency/update rate at which the data file was recorded. This information is contained within the
        non-tabular part of the data file and is located by finding the line containing the search phrase (name_in_file)
        and retrieving the numerical part from this.
        :return: the frequency (float)
        """
        with open(self.file_path, "r") as f:
            update_rate_line = f.read().strip()
            update_rate = float(update_rate_line[update_rate_line.rfind(name_in_file) + len(name_in_file):update_rate_line.rfind('Hz')])
        return update_rate

    def get_data_from_file(self, attributes):
        """
        Gets data from a data file. Multidimensional attributes are grouped together. So to get 3 dimensional
        Acceleration (specified by Acc_X, Acc_Y, Acc_Z) in the data file 'Acc' should be passed as one of the data
        attributes, similarly this can be done for any multidimensional attribute specified in this way. Angles is used
        for Roll, Pitch and Yaw.
        :param attributes: List of attributes
        :return: a dictionary mapping from the attribute to a numpy array of shape (num_frame, num_dimensions) of the
                 data for that attribute.
        """
        df = pd.read_table(self.file_path, header=self.header_rows)

        data = {}
        for attribute in attributes:
            if attribute in df:
                # Sometimes when reading in from data files the values read are off by one (in the last decimal place).
                # Seeing as data in data files is recorded to 6 decimal places rounding to 6 decimal places corrects this
                # error
                data[attribute] = np.array(list(map(lambda x: round(x, 6), df[attribute])))
            else:
                # Group multidimensional data together
                # example Acc_X, Acc_Y and Acc_Z get grouped into a 3 dimensional list of lists ([Acc_X, Acc_Y, Acc_Z]) and
                # this with the key "Acc"
                if attribute is 'Angles':
                    suffixes = ["Roll", "Pitch", "Yaw"]

                else:
                    suffixes = ["_X", "_Y", "_Z"]

                values = []
                for suffix in suffixes:
                    if attribute is 'Angles':
                        values.append(list(map(lambda x: round(x, 6), df[suffix])))
                    else:
                        values.append(list(map(lambda x: round(x, 6), df[attribute + suffix])))

                # make shape (num_frame, num_dim) so that for each multidimensional attribute (x, y, z) coordinates are
                # the lowest sub array.
                data[attribute] = np.array(values).T

        return data