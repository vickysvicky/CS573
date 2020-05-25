README

main.py

----------------------PROGRAM DESCRIPTION-----------------------

This program allows user to compare neural network classifiers
performance with different neural network and different
hyperparameters on Optical Recognition of Handwritten Digits Data
Set from UC Irvine Machine Learning Repository. It uses python's
tensorflow.keras library to construct the classifiers.

Users will be able to choose between what elements to compare
with, and manually set the value of the hyperparameters. The
hyperparameter that can be configured by users are: learning
rate, momentum, number of training epoch, number of hidden
layers, and number of nodes (hidden size).

Program will display the overall classification accuracy, class
accuracy, and confusion matrix for both training and testing data.

Users are able to choose between <1> fully-connected feed forward
neural network, or <2> convolutional network (CNN).

In <1> fully-connected feed forward network, users can choose to
compare between:
        <1> Error function: Mean-Squared-Error (MSE) VS Cross-
                            Entropy Error
        <2> Hidden units:   tanh VS ReLU

In <1.1>, ReLU is used as the default hidden units and cannot be
changed. In <1.2>, cross-entropy error is use as the default
error function and cannot be changed.

In <2> CNN, users can set the hyperparameters and see the results.


----------------------------HOW TO-------------------------------

Prior to running the program, user need to install some packages
by running the following command:

    pip install -r requirements.txt

Then, the program can be run with the line below:

	python main.py


