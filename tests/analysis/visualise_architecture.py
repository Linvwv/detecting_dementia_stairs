"""
This program creates the log files necessary for TensorBoard to visualise the architecture. It provides code for
changing the parameters of the model, the number of windows, number of features and number of classes. Different
versions were experimented with to ensure the model worked properly
In order to display the visualisation follow the steps below. The logfiles are stored in a folder called tmp in the
directory containing this python file.
1 - Go to the terminal
2 - Use the tensorboard command to display the visualisation on a local server. If the path to the tmp directory created
    is path/to/tmp/ then run the following command -
    tensorboard --logdir=path/to/tmp/ --port 6006
3 - Open the following link in your browser - http://localhost:6006/
"""
import sys, os

sys.path.append(os.path.join(sys.path[0], "../.."))

from analysis.rnn_model import RNNModel
import settings

settings.num_windows = settings.max_reading_length

num_classes = 3
num_features = len(settings.used_sensors) * len(settings.used_sensor_attributes) * settings.num_windows
clf_params = None

model = RNNModel(num_classes, num_features, clf_params=clf_params)

model.spawn_clf()
model.write_graph_for_visualisation()
