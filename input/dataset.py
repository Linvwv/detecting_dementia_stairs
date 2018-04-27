import pandas as pd
from os import path
import settings


class DataSet():
    """
    A class encapsulating the Data Set consisting of data from each trial for every participant. Along with methods for
    manipulation of this data such as insertion, removal, etc. This class provides a wrapper around Pandas DataFrame
    object that is used to store and represent the data in a tabular form, with columns as a data attribute and rows as
    the corresponding data attributes for a particular trial. NOTE: row number 0 is the row of the first trial, while
    this may be row 2 in the excel file.
    """
    def __init__(self, data_dir=settings.data_dir, file_name=settings.lookup_file, sheet=settings.sheet):
        """
        The constructor for the DataSet object
        :param data_dir: The directory in which the look up is located. By default this is taken from settings.py
        :param file_name: The name of the excel file containing the data. In this project it was the lookup file. By
                          default this is taken from settings.py
        :param sheet: The sheet of the file to be used, the task that is being analysed. For this project it was the
                      "Stairs" sheet. By default this is taken from settings.py
        """
        self.data_dir = data_dir
        self.file_name = file_name
        self.sheet = sheet
        self.data = None

    def setup(self, ignore_trials=None):
        """
        Reads the data from the file and does some basic manipulations such as removing entries with missing values,
        removing corrupted trials. Corrupted trials are identified manually and are provided to this method as a
        list of their row numbers (ignore_trials). It also calculates the paths to the datafiles for the rest of the
        trials and adds this to the dataset, dropping the attributes used to build these paths.
        :param ignore_trials: Corrupted trials to be ignored.
        """
        self.read_dataset()
        if ignore_trials is not None:
            self.remove_problem_trials(remove_rows=ignore_trials)
        self.calc_data_files()
        self.remove_columns(settings.unused_columns)

    def read_dataset(self):
        """
        Reads dataset (lookup file) as a Pandas DataFrame.
        """
        self.data = pd.read_excel(path.join(self.data_dir, self.file_name), sheet_name=self.sheet)

    def remove_empty_rows(self, check_columns=None, reset_index = True):
        """
        Removes rows with missing entries in the columns specified. If no columns are specified then those that are
        needed to build the datafile paths are used and other columns which have missing data that is needed for the
        analyses conducted.
        :param check_columns: Column names for columns to be checked for missing entries. If none then the columns
                              containing attributes that make up te data files are checked along with other columns that
                              are needed for the analysis that have been identified as having missing values.
        """
        if check_columns is None:
            check_columns = ["DISTANCETRIAL", "LF Sensor No.", "RF Sensor No.", "P Sensor No.", "File name root"]

        for column in check_columns:
            self.data = self.data[self.data[column].notnull()]

        if reset_index:
            self.data = self.data.reset_index(drop=True)

    def remove_rows(self, remove_rows):
        """
        Removes rows specified by remove_rows, used for removing corrupted trials. Note this method does not reset the
        indices therefore, row numbers after removal remain the same. These must be reset manually after calling this
        method or a wrapper function for this method should be used
        :param remove_rows: a single row or list of rows to be removed as an index starting from 0
        """
        self.data = self.data.drop(remove_rows)

    def remove_columns(self, columns):
        """
        Removes columns from the datafile, used to get rid of unused or redundant data so that the datafile does not
        grow uncontrolled.
        :param columns: A single column or list of columns.
        """
        columns = list(filter(lambda x: x in self.data.columns, columns))
        self.data = self.data.drop(columns, axis=1)

    def remove_problem_trials(self, remove_rows, check_columns=None):
        """
        Removes trials that cannot be used this includes corrupted trials and trials where data is not present.
        :param remove_rows: rows to be removed as an index starting from 0, used for identifying corrupted trials.
        :param check_columns: columns to be checked for missing entries. These are columns where the data is needed for
                              conducting analysis.
        """
        self.remove_rows(remove_rows)
        self.remove_empty_rows(check_columns=check_columns, reset_index=False)

        # reset indices of dropped rows are removed
        self.data = self.data.reset_index(drop=True)

    def calc_data_files(self):
        """
        Calculates the data file for each sensor for each trial and stores this in the dataset with a column for each
        sensor. It also removes the columns of data that were used for creating the file path as these are no longer
        needed.
        """
        for prefix in ["LF", "RF", "P"]:
            # calculate file path
            base = self.data_dir + self.data["Day Code"] + "_Exported/"
            files = base + self.data["File name root"] + self.data[prefix + " Sensor No."] + ".txt"

            # check if this version of the path exists for each file and replace with the other version if it does not
            dont_exist = [not path.isfile(file) for file in files]
            files[dont_exist] = base + self.data['Day Code'] + "_" + self.data["File name root"] + \
                                self.data[prefix + " Sensor No."] + ".txt"
            self.data[prefix.lower() + "_sensor"] = files

        self.remove_columns(["File name root", "LF Sensor No.", "RF Sensor No.", "P Sensor No."])

    def get_num_trials(self):
        """
        :return: The number of trials/rows in the data set, the size of the dataset.
        """
        return self.data.shape[0]

    def get_data_attribute(self, columns, row=None):
        """
        Gets data attribute(s) from the dataset. Can get all rows if no row is specified or a particular row
        :param columns: Single column or multiple columns, specified as a list
        :param row: row number (starting at 0) or None for all rows
        :return: The data attribute as a Series (for multiple rows) or as a single value (single rows)
        """
        if row is None:
            return self.data[columns]
        else:
            return self.data[columns][row]

    def set_data_attributes(self, data_frame):
        """
        Adds in data for attributes given by a dataframe. It joins them together by concatenatting them on the axis with
        columns.
        :param data_frame: The data_frame to be joined to the dataset.
        """
        self.data = pd.concat([self.data, data_frame], axis=1)

    def get_data_files(self, row):
        """
        Gets the data files for each sensor for a particular trial specified by the row
        :param row: row number (starting at 0)
        """
        data_files = {}
        for sensor in ["lf_sensor", "rf_sensor", "p_sensor"]:
            data_files[sensor] = self.get_data_attribute(sensor, row)

        return data_files
