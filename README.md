# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This project is about optimizing a machine learning pipeline using a example dataset containinig data about clients churn in banking. We will try to predict the churn tendency of the different clients using the features present in the dataset.

For this purpose we will use two different approaches, the first one by a simple classification model using a logistic regression algorithm that is tuned by hyperdrive and the second approach is using AutoML. 

## Scikit-learn Pipeline

The frontend for all the interactions with the different part of the model is a Jupyter Notebook. On the backend we have a train.py script that does all the heavy work from a machine learning point of view, this scripts preprocess all the data and builds a dataset that has all the categorical variables transformed by one-hot encoding, all the dates are transformed to numerical format, inside the script the data is also splitted in train and test sets and finally a logistic regression algorithm is trained using two model parameters that are passed as arguments and accuracy metric is obtained.

This scripts with its arguments is used by an hyperdrive optimizer that tries to find the best hyperparameters for that model. Once this best hyperparameters are found, we train a logistic regression model with those hyperparameters and we save it.

For the hyperdrive hyperparameter search we use a random sampler, this has the advantage of producing good results without taking too much computational time.

We also implemented an early stopping policy for the hyperparameters optimization. Using an early stopping policy with this kind of optimal hyperparameters search tools like hyperdrive has the advantage of getting fastest results because if the algorithm detects that the current points selected in the hyperparameters search space are not going to improve previous results it stops the iterations and try another set of points in the hyperparameters search space. If an early stopping policy is not implemented the optimizer will continue the calculations until convergence is reached for every set of points selected for the hyperparameter search, not taking into account if they are improving or not the metric of the (in this case) classification algorithm. 

## AutoML

The model generated after running the AutoML pipeline was a VotingEnsembled model, this model is built by a StandardScaler for preprocessing the numerical features and a XGBoost algorithm for the classification.

## Pipeline comparison

Both models have similar perfomance if we look at the main metric used: accuracy, both reaching values in the 0.91... what is a very high value for a classification if we assume that we have a balanced dataset for the target label.

We have chosen to feed the models with the same preprocessed data (one hot encoding and the dates transformed to numerical) but inside the AutoML model architecture some additional steps in the preprocessing phase are taken, as scaling of numerical data. This can help the models to get information from the data more easily.

The Scikit-learn model is a very simple one, it is a Logistic Regression, so its explicability is very high just looking at the coefficients of the equation for the logistic regression, in the other hand the best AutoML model is an ensemble model based on multiple decision trees as XGBoost, this kind of model is more difficult to interpret by the user but it usually provides more perfomance (more accuracy), but in this case, as we stated before, the difference in perfomance is negligible, so in my opinion it is better to choose the simplest model because of the best explicability and because we will probably need less computation time.

## Future work

For the Scikit-learn model would be needed to try different hyperparameters to tune the model a little bit more. And maybe try some scaling techniques for the numerical data as a improvement in the preprocessing phase.

For the AutoML we could increment the total number of iterations so we could check for additional combinations of different algorithms and hyperparameters for them.

Finally I would also try different metrics to check the results obtained so far and to improve the behaviour of the models facing the case of an unbalanced dataset.
