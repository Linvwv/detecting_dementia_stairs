# Detecting Dementia Using Gait on Stairs
This repository contains all the code used for running the analyses presented in the paper "Automating assessment of the risk of dementia and dementia phenotype by analysing the gait of people with Posterior Cortical Atrophy (PCA) and typical Alzheimer’s Disease (tAD) while they traverse a staircase", which was based on the dissertation submitted by William Bhot (the second author of the paper) under the supervision of Catherine Holloway (the first author) as part of an MEng in Computer Science at University College London. For details about the analyses and results please read the paper. Information about running the analyses presented in the paper can be found below.

To run any analysis the models need to set up. See the following section for instructions on how to do this. Note that the analysis on the importance of sensors is in a different file (`main_sensors.py`) from all the other analyses, however, the instructions below still apply as the two files mirror each other. After the model is set up choose which of the analyses to run and follow the steps in the appropriate sections.

## Setting up the models
1. Navigate to `main.py` in the root of the repository.
2. Set the parameters you would like to run the analysis with (lines 20-30). This allows you to set parameters for the analysis such as which sensors to use, how many windows to use when constructing the features and what data attributes to use, but it also lets you set up parameters specific to your environment and code setup such as the directory containing the data. This directory must contain a lookup file and all the data must be contained within sub-directories. For more information on the parameters for the analysis see the paper.
3. Choose the parameters for the model by uncommenting them and commenting out the parameters for all other models (lines 37-97). Alternatively you can use your own parameters.
4. Create the appropriate model. This involves choosing whether to use a binary model (for dementia detection) or a multiclass model (for type detection). Furthermore, if an RNN model was chosen in the previous step then the RNN variant of the model should be chosen (ie `RNNModel`) otherwise the `ClassicalModel` should be chosen. Choosing the correct model just involves uncommenting the right line of code (lines 106-114).

## Setting up the analysis
### Importance of sensors
1. To run this analysis navigate to `main_sensors.py` in the root of the repository.
2. Follow step 2 from Setting up the models.
3. Run this file.

### Predicting dementia and type
1. To run this analysis follow all the steps in Setting up the models
2. Uncomment line 126 to 129.
3. Run this file.

NOTE: For confusion matrices uncomment lines 132-137 instead of step 2

### Importance of features
1. To run this analysis follow all the steps in Setting up the models.
2. Uncomment line 145-146.
3. Run this file.
