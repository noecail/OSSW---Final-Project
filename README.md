# OSSW - Final-Project
# Noé Caillet - 50231569

## Motivation
The building of this model is the purpose of the final project of the OpenSource Software class at Chung-Ang University.
The project was made during december, 2023
Author : Noé Caillet

This project is about finding the best model to fit a set of training data points.
The model is trained with dataset of four different tumor types and will be tested on a
test dataset that is unknown at the time of the training phase.

## Libraries used
List of the Python libraries used for this project : OS, Scikit-learn, Scikit-image, Numpy, Matplotlib.

## Making process
(Refer to the bottom of this file)*

## The model itself
> rfc = sklearn.ensemble.RandomForestClassifier(random_state=55, bootstrap=False, max_depth=83, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=270)


## Usage
To use the Jupyter Notebook file, first run all the cells then use the trained model on you test dataset.

## License
GNU

*## Making process
The dataset consists of four folders each containing a few hundreds of jpg pictures
of tumored or healthy brains.

I chose the random forest classifier algorithm for its high accuracy and 
its emphasize on feature importance. This is highly important in our case because the
number of features in our dataset is huge. Thus, I have to consider only the
ones that have the most impact on the prediction. The selection is made using the
attribute feature_importances_ and in the end, I only consider 36 features of our
dataset, which is a big gain of time.

This being said, the process of choosing the hyperparameters comes.

1) First approach with RandomSearch & CrossValidation
The idea with RandomSearch is to train different models with hyperparameter values varying
within a range of set values. The algorithm then returns the values that produce the model
with the highest accuracy.
The CrossValidation is an approach to avoid the issue of overfitting. A model trained to
a single dataset can become very specific to this dataset and prove itslef inaccurate when
tested. The CrossValidation approach further splits multiple times our training datasets
into smaller training and testing datasets, each time with different parts of the original
dataset, so that the final model training is validated by crossing all smaller trainings.

This first part gave me values which to further study around.

2) Second approach with GridSearch & CrossValidation
This is almost the same process as before, except the values succesively taken by the
hyperparameters are previously determined and all of them will be used throughout the
training process.
The idea here is to get more precise on the hyperparameters values I got from the first
step.

3)At the very end, I tried different values of random_parameter. The one I chose gave me
an accuracy of 0.92 on the given dataset.
