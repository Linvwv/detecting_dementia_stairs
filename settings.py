# The maximum length of readings or number of frames in a trial after clipping
max_reading_length = 1450

# the number of windows to use for the windowed average
num_windows = max_reading_length

# Dataset parameter
sheet = "Stairs"
ignore_trials = [36, 37, 40, 41, 42, 43, 44, 45, 189, 376, 415]

# sensor constants - used_sensors are the ones used for the analysis
all_sensors = ["lf_sensor", "rf_sensor", "p_sensor"]
foot_sensors = ["lf_sensor", "rf_sensor"]
used_sensors = foot_sensors

# the sensor attributes used for the analysis
used_sensor_attributes = ["Acc", "Gyr"]

# the directory in which the datafiles are contained
data_dir = 'data/'
# the lookup file name
lookup_file = 'SWTS1_Xsens_Look_Up.xlsx'
# columns of the lookup file that are not used by the analysis
unused_columns = ['Participant number', 'Trial', 'Order', 'Block', 'Randomisation schedule', 'Cue (0/1)',
                  'Lighting (1/2)', 'Easy  (1/2/3/4)', 'Holding bannister (0/1)?', 'Age', 'DD', 'Female']