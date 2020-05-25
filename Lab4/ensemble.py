##
#   thanks to https://scikit-learn.org/stable/modules/ensemble.html
##

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#   to suppress Convergence warning and some other Users warning
import warnings
warnings.filterwarnings("ignore")

#   get data
train_data = pd.read_csv('lab4-train.csv', delimiter=',', dtype=np.int64)
test_data = pd.read_csv('lab4-test.csv', delimiter=',', dtype=np.int64)
#   last column is class
train_x, train_y = train_data.loc[:, ['R', 'F', 'M', 'T']], train_data.loc[:, ['Class']]
test_x, test_y = test_data.loc[:, ['R', 'F', 'M', 'T']], test_data.loc[:, ['Class']]

#   TASK 1
#   train random forest and adaboost
#   experiment hyperparams of models (number of base classifiers, etc)
#   report overall accuracy and confusion matrix
#   random_state is set to 1 for reproducibility
print("\t TASK 1\n")


#   ADABOOST
model_ada = AdaBoostClassifier(random_state=1)
#   find best hyperparam
param = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.95, 1], 'n_estimators': [1, 2, 3, 4, 8, 16, 32]}
grid_ada = GridSearchCV(model_ada, param)
grid_ada.fit(train_x, train_y.values.ravel())
print("Best hyperparam for AdaBoost: ", grid_ada.best_estimator_.get_params())
#   report accuracy
pred_y_ada = grid_ada.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_ada))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_ada))
pred_y_ada = grid_ada.predict(test_x)
print("Accuracy for test data: ", accuracy_score(test_y.values.ravel(), pred_y_ada))
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_ada))

#   RANDOMFOREST
model_forest = RandomForestClassifier(random_state=1)
#   find best hyperparam
param = {'n_estimators': [1, 2, 3, 4, 8, 16, 32], 'max_depth': [1, 5, 10, 20, None], 'min_samples_leaf': [1, 2, 4, 8]}
grid_forest = GridSearchCV(model_forest, param)
grid_forest.fit(train_x, train_y.values.ravel())
print("Best hyperparam for RandomForest: ", grid_forest.best_estimator_.get_params())
#   report accuracy
pred_y_forest = grid_forest.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_forest))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_forest))
pred_y_forest = grid_forest.predict(test_x)
print("Accuracy for test data: ", accuracy_score(test_y.values.ravel(), pred_y_forest))
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_forest))


#   TASK 2
#   train neural network, logistic regression, naive bayes, decision tree
#   slightly tune hyperparams
#   report confusion matrix and classification accuracy
#   construct ensemble with unweighted majority vote
#   construct ensemble with weighted majority vote
#   report performance
print("\n\n\t TASK 2 \n")


#   NEURAL NETWORK - MLP
model_nn = MLPClassifier(random_state=1)
#   find best param
param = {'hidden_layer_sizes': [1, 2, 4, 8, 16, 32], 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
         'learning_rate': ['constant', 'invscaling', 'adaptive']}
grid_nn = GridSearchCV(model_nn, param)
grid_nn.fit(train_x, train_y.values.ravel())
print("Best hyperparam for Neural Network: ", grid_nn.best_estimator_.get_params())
#   report accuracy
pred_y_nn = grid_nn.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_nn))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_nn))
pred_y_nn = grid_nn.predict(test_x)
score_nn = accuracy_score(test_y.values.ravel(), pred_y_nn)
print("Accuracy for test data: ", score_nn)
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_nn))


#   LOGISTIC REGRESSION
model_lr = LogisticRegression(random_state=1)
#   find best param
param = {'penalty': ['l2', 'none'], 'C': [0.01, 0.1, 0.5, 1, 2], 'solver': ['lbfgs', 'sag', 'saga']}
grid_lr = GridSearchCV(model_lr, param)
grid_lr.fit(train_x, train_y.values.ravel())
print("Best hyperparam for Logistic Regression: ", grid_lr.best_estimator_.get_params())
#   report accuracy
pred_y_lr = grid_lr.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_lr))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_lr))
pred_y_lr = grid_lr.predict(test_x)
score_lr = accuracy_score(test_y.values.ravel(), pred_y_lr)
print("Accuracy for test data: ", score_lr)
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_lr))


#   NAIVE BAYES
model_nb = GaussianNB()
#   find best param
param = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}
grid_nb = GridSearchCV(model_nb, param)
grid_nb.fit(train_x, train_y.values.ravel())
print("Best hyperparam for Naive Bayes: ", grid_nb.best_estimator_.get_params())
#   report accuracy
pred_y_nb = grid_nb.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_nb))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_nb))
pred_y_nb = grid_nb.predict(test_x)
score_nb = accuracy_score(test_y.values.ravel(), pred_y_nb)
print("Accuracy for test data: ", score_nb)
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_nb))


#   DECISION TREE
model_tree = DecisionTreeClassifier(random_state=1)
#   find best param
param = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [1, 5, 10, 20, None],
         'min_samples_leaf': [1, 2, 4, 8]}
grid_tree = GridSearchCV(model_tree, param)
grid_tree.fit(train_x, train_y.values.ravel())
print("Best hyperparam for Decision Tree: ", grid_tree.best_estimator_.get_params())
#   report accuracy
pred_y_tree = grid_tree.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_tree))
print("Confusion matrix: \n", confusion_matrix(train_y, pred_y_tree))
pred_y_tree = grid_tree.predict(test_x)
score_tree = accuracy_score(test_y.values.ravel(), pred_y_tree)
print("Accuracy for test data: ", score_tree)
print("Confusion matrix: \n", confusion_matrix(test_y, pred_y_tree))


#   UNWEIGHTED MAJORITY VOTE
est = [('nn', grid_nn.best_estimator_), ('lr', grid_lr.best_estimator_), ('nb', grid_nb.best_estimator_),
       ('tree', grid_tree.best_estimator_)]
vote_xweight = VotingClassifier(estimators=est, voting='hard', weights=[1, 1, 1, 1])
vote_xweight.fit(train_x, train_y.values.ravel())
print("\nUnweighted majority vote")
pred_y_vxw = vote_xweight.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_vxw))
pred_y_vxw = vote_xweight.predict(test_x)
print("Accuracy for test data: ", accuracy_score(pred_y_vxw, test_y.values.ravel()))


#   WEIGHTED MAJORITY RULE
vote_weight = VotingClassifier(estimators=est, voting='hard', weights=[score_nn, score_lr, score_nb, score_tree])
vote_weight.fit(train_x, train_y.values.ravel())
pred_y_vw = vote_weight.predict(train_x)
print("\nWeighted(w accuracy) majority vote")
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_vw))
pred_y_vw = vote_weight.predict(test_x)
print("Accuracy for test data: ", accuracy_score(test_y.values.ravel(), pred_y_vw))

#   WEIGHTED MAJORITY RULE
vote_weight = VotingClassifier(estimators=est, voting='hard', weights=[1, 1, 4, 1])
vote_weight.fit(train_x, train_y.values.ravel())
print("\nWeighted(NB is major) majority vote")
pred_y_vw = vote_weight.predict(train_x)
print("Accuracy for train data: ", accuracy_score(train_y.values.ravel(), pred_y_vw))
pred_y_vw = vote_weight.predict(test_x)
print("Accuracy for test data: ", accuracy_score(test_y.values.ravel(), pred_y_vw))

# #   find best param
# param = {'weights': [[score_nn, score_lr, score_nb, score_tree], [4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 4, 1], [1, 1, 1, 4]]}
# grid_vote = GridSearchCV(vote_weight, param)
# grid_vote.fit(train_x, train_y.values.ravel())
# # print("Best weights: ", grid_vote.best_estimator_.get_params())
# #   report accuracy
# pred_y_vg = grid_vote.predict(test_x)
# print("Accuracy with grid search: ", accuracy_score(pred_y_vg, test_y.values.ravel()))

