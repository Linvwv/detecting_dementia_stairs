"""
This file is not used by the current system as gait parameters (as they were calculated) were found to be uninformative
and actually reduced the score of classifiers, potentially due to overfitting to the training data. However, it was
included for reference.
"""


def get_swing_times(foot_swing_indices, frequency=50.0):
    """
    Calculates the swing times of each step taken by a single foot based on the foot swing indices.
    :param foot_swing_indices: A numpy array containing 2 sub-arrays containing the start indices and end indices of
                               each swing phase respectively (for a single foot). The shape of this is (2, num_steps).
    :param frequency: The frequency of the trial, set to 50.0 by default. All trials analysed by this project used this
                      frequency as checked by an assertion. A float.
    :return: A numpy array containing the swing times of each step on a single foot. Shape (num_steps,)
    """
    num_swing_frames = foot_swing_indices[1] - foot_swing_indices[0]
    return num_swing_frames / frequency


def get_stance_times(foot_swing_indices, frequency=50.0):
    """
    Calculates the stance times for each step taken by a single foot based on the foot swing indices.
    :param foot_swing_indices: A numpy array containing 2 sub-arrays containing the start indices and end indices of
                               each swing phase respectively (for a single foot). The shape of this is (2, num_steps).
    :param frequency: The frequency of the trial, set to 50.0 by default. All trials analysed by this project used this
                      frequency as checked by an assertion. A float.
    :return: A numpy array containing the swing times of each step on a single foot. Shape (num_steps,)
    """
    num_stance_frames = foot_swing_indices[0][1:] - foot_swing_indices[1][:-1]
    return num_stance_frames / frequency


def get_num_steps(trial_swing_indices):
    """
    Gets the number of steps (across both feet) in a single trial. As given by the trial_swing_indices.
    :param trial_swing_indices: The swing indices of the trial. A dictionary mapping from sensor name (lf_sensor or
                                rf_sensor) to the foot_swing_indices - a 2d numpy array with shape
                                (2, num_steps_with_foot)
    :return: The number of steps in a single trial across both feet.
    """
    return len(trial_swing_indices["lf_sensor"][0]) + len(trial_swing_indices["rf_sensor"][0])

def get_cadence(num_steps, trial_time):
    """
    Gets the cadence of a single trial given the number of steps and the time that a trial takes. Cadence is defined as
    the number of steps per minute.
    :param num_steps: Number of steps.
    :param trial_time: Number of seconds in the trial.
    :return: The cadence, a float. Defined as - number of steps per minute.
    """
    if trial_time == 0:
        raise ZeroDivisionError("Trial time cannot be 0")
    return num_steps / trial_time * 60

