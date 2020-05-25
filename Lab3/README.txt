README

decTree.py

----------------------PROGRAM DESCRIPTION-----------------------

This program uses python's sklearn.tree library to construct
Decision Tree Classifier to classify the Congressional Voting 
Records data set from UC Irvine Machine Learning Repository.

A 5-fold cross validation is then carried out on the model
learned from all the data on the data set.

An experiment to randomly split the dataset into 5 roughly same
sized datasets, train the model with 4 of the datasets, and test
its performance on the remaining 1 dataset.

The output of this program is just the accuracy of the model from 
the 5-fold cross validation and the experimental splits, and
figures of the each of the decision trees.


----------------------------HOW TO-------------------------------

Prior to running the program, user need to install some packages
by running the following command:

    pip install -r requirements.txt

Then, the program can be run with the line below:

	python decTree.py


