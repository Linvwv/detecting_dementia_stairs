"""
Utilities for conducting input such as restructuring the format of the input data and wrappper functions for reading
data files.
"""

from input.data_file import DataFile


def bring_attributes_up(data):
    """
    Data retrieved from files is structured as a dictionary with top level keys being the sensor and second level keys
    being the sensor attribute. However, for processing dictionaries with top level keys being the attribute and second
    level keys being the sensor are used. This method converts from the first format to the second (this could be
    viewed as analogous to transposing a matrix (but for dictionaries).
    :param data: a dictionary of the input data, stored in the first format mentioned above
    :return: a dictionary of the input data, stored in the second format mentioned above
    """
    ret = {}
    for sensor in data:
        for attribute in data[sensor]:
            if attribute in ret:
                ret[attribute][sensor] = data[sensor][attribute]
            else:
                ret[attribute] = {sensor: data[sensor][attribute]}
    return ret


def read_trial_data(data_files, sensors, attributes):
    """
    A wrapper function that reads the data files for a single trial. ie reads the data file for each sensor and gets
    the attribute. It also checks the frequency to make sure the assumption that all files are recorded at 50.0Hz is
    true and reformats the input dictionary read from the file so the attributes are the top level keys
    (see bring_attributes_up above)
    :param data_files: a dictionary mapping from the name of the sensor to the path of the data file for that sensor
    :param sensors: The names of the sensors
    :param attributes: The attributes to retrieve from the data file, such as Acc, Gyr, Angles, etc.
    :return: a dictionary mapping from attribute names to a sub-dictionary in turn mapping from sensor names to the
             value of the attribute for that particular sensor.
    """
    data = {}

    for sensor in sensors:
        file = DataFile(data_files[sensor])
        # if a file is found with update rate that is not 50.0 this needs to be taken into account
        update_rate = file.get_frequency()
        assert update_rate == 50.0

        data[sensor] = file.get_data_from_file(attributes)

    return bring_attributes_up(data)
