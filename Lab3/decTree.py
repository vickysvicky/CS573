import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
import graphviz

debug = False

#   read in data
col = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
          'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
          'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
          'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
          'export-administration-act-south-africa']
data = pd.read_csv('house-votes-84.data', header=None)
#   preprocessing
data = data.replace('?', np.NaN)
data = data.replace('y', 1)
data = data.replace('n', 0)
print("Total number of data: " + str(len(data)))
#   first attr is actual class
train_X, train_Y = data.loc[:, 1:], data.loc[:, 0]
#   replace missing values with mean
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp = imp.fit(train_X)
train_X_imp = imp.transform(train_X)


#   build model
print("Train with all data...")
model = tree.DecisionTreeClassifier()
model.fit(train_X_imp, train_Y)

#   visualize tree
dectree = tree.export_graphviz(model, filled=True, feature_names=col[1:], class_names=model.classes_, rounded=True)
dec3 = graphviz.Source(dectree)
dec3.render("tree_viz", format='png')

#   5 fold cross validation
score = cross_val_score(model, train_X_imp, train_Y, cv=5)
# print(score)
print("5 fold cross validation")
print("Accuracy (95% confidence interval): " + str(score.mean()*100) + "\t(+/- " + str(2*score.std()*100) + ")")
print()


#   Experiment
#   randomly split into 5 dataset
print("Randomly split dataset into 5 datasets and train with four of the sets and test with the remaining one each iteration")
kf = KFold(n_splits=5, shuffle=True, random_state=0)    # random state set to 1 for repoducibility
split = 1
# print(kf.split(train_X_imp))
for train_idx, test_idx in kf.split(train_X_imp):
    #   get split data
    train_x_split, train_y_split = train_X_imp[train_idx], train_Y[train_idx]
    test_x_split, test_y_split = train_X_imp[test_idx], train_Y[test_idx]
    #   learn model
    model.fit(train_x_split, train_y_split)
    #   visualize tree
    dectree = tree.export_graphviz(model, filled=True, feature_names=col[1:], class_names=model.classes_, rounded=True)
    dec3 = graphviz.Source(dectree)
    dec3.render('split%d_train_tree' % split, format='png')
    #   test model and get accuracy
    test_y_pred = model.predict(test_x_split)
    if debug is True:
        print("Split %d" % split)
        print("Test index: " + str(test_idx))
    print("Accuracy for split %d: " % split + str(accuracy_score(test_y_pred, test_y_split)*100))
    split += 1
