Sample scripts to remove the noise from a neutron reflection profile
====

This is the demonstration python script to reduce the statistical noise from a neutron reflection profile. The detailed discussion is given in the paper, "Deep learning approach for interface structure analysis with large statistical noise in neutron reflectometry" by H. Aoki, Y. Liu, T. Yamashita.

## Usage
This script is coded on Python 3.8.5 with Pytorch 1.8.1. For the execution of this script, Python, Pytorch, and the other requirements should be installed. The sample data sets of the simulated and ground truth NR profile for the training, "training_data.csv" and "training_target.csv", respectively, are included. The number of the simulated and ground truth NR profile is 4096. The NR profiles at 0.3, 0.7, 1.6, and 3.5 degrees are concatenated to a single one-dimensional array. The data in "test_data.csv" and "test_target.csv" are the simulated and ground truth NR profiles, respectively, which was separately generated from those for the training data set. They are used for the demonstration of the data processing by the trained model.

### Traning
Execute the following command. The training is conducted by the data files of "training_data.csv" and "training_target.csv".
```
python train.py
```
The trained model is saved as "model.pth".

### Prediction
Execute the following command. The prediction of the NR profile is performed for a single NR profile selected from "test_data.csv" randomly.
```
python predict.py
```
It outputs a PNG file ("out.png") to show the plot of the simulated, predicted, and corresponding ground truth profiles.
