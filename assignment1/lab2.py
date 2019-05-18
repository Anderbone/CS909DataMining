# %matplotlib inline

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

import pydot, io
import time

#######################End imports###################################
####################Do not change anything below
# Load MNIST data. fetch_mldata will download the dataset and put it in a folder called mldata.
# Some things to be aware of:
#   The folder mldata will be created in the folder in which you started the notebook
#   So to make your life easy, always start IPython notebook from same folder.
#   Else the following code will keep downloading MNIST data
try:
    mnist = fetch_mldata("MNIST original")

except Exception as ex:
    import tensorflow.examples.tutorials.mnist.input_data as input_data

    m = input_data.read_data_sets("MNIST")
    data = np.concatenate((m.train.images, m.test.images))
    target = np.concatenate((m.train.labels, m.test.labels))


    class dataFrame:
        def __init__(self, data, target):
            self.data = data
            self.target = target


    mnist = dataFrame(data, target)

# The data is organized as follows:
#  Each row corresponds to an image
#  Each image has 28*28 pixels which is then linearized to a vector of size 784 (ie. 28*28)
# mnist.data gives the image information while mnist.target gives the number in the image
print("#Images = %d and #Pixel per image = %s" % (mnist.data.shape[0], mnist.data.shape[1]))

# Print first row of the dataset
img = mnist.data[0]
print("First image shows %d" % (mnist.target[0]))
print("The corresponding matrix version of image is \n", img)
print("The image in grey shape is ")
plt.imshow(img.reshape(28, 28), cmap="Greys")

# First 60K images are for training and last 10K are for testing
all_train_data = mnist.data[:60000]
all_test_data = mnist.data[60000:]
all_train_labels = mnist.target[:60000]
all_test_labels = mnist.target[60000:]

# For the first task, we will be doing binary classification and focus  on two pairs of
#  numbers: 7 and 9 which are known to be hard to distinguish
# Get all the seven images
sevens_data = mnist.data[mnist.target == 7]
# Get all the none images
nines_data = mnist.data[mnist.target == 9]
# Merge them to create a new dataset
binary_class_data = np.vstack([sevens_data, nines_data])
binary_class_labels = np.hstack([np.repeat(7, sevens_data.shape[0]), np.repeat(9, nines_data.shape[0])])

# In order to make the experiments repeatable, we will seed the random number generator to a known value
# That way the results of the experiments will always be same
np.random.seed(1234)
# randomly shuffle the data
binary_class_data, binary_class_labels = shuffle(binary_class_data, binary_class_labels)
print("Shape of data and labels are :", binary_class_data.shape, binary_class_labels.shape)

# There are approximately 14K images of 7 and 9.
# Let us take the first 5000 as training and remaining as test data
orig_binary_class_training_data = binary_class_data[:5000]
binary_class_training_labels = binary_class_labels[:5000]
orig_binary_class_testing_data = binary_class_data[5000:]
binary_class_testing_labels = binary_class_labels[5000:]

# The images are in grey scale where each number is between 0 to 255
# Now let us normalize them so that the values are between 0 and 1.
# This will be the only modification we will make to the image
binary_class_training_data = orig_binary_class_training_data / 255.0
binary_class_testing_data = orig_binary_class_testing_data / 255.0
scaled_training_data = all_train_data / 255.0
scaled_testing_data = all_test_data / 255.0

print(binary_class_training_data[0, :])

###########Make sure that you remember the variable names and their meaning
# binary_class_training_data, binary_class_training_labels: Normalized images of 7 and 9 and the correct labels for training
# binary_class_testing_data, binary_class_testing_labels : Normalized images of 7 and 9 and correct labels for testing
# orig_binary_class_training_data, orig_binary_class_testing_data: Unnormalized images of 7 and 9
# all_train_data, all_test_data: un normalized images of all digits
# all_train_labels, all_test_labels: labels for all digits
# scaled_training_data, scaled_testing_data: Normalized version of all_train_data, all_test_data for all digits
###Do not make any change below
def plot_dtree(model,fileName):
    #You would have to install a Python package pydot
    #You would also have to install graphviz for your system - see http://www.graphviz.org/Download..php
    #If you get any pydot error, see url
    # http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will
    dot_tree_data = io.StringIO()
    tree.export_graphviz(model, out_file = dot_tree_data)
    (dtree_graph,) = pydot.graph_from_dot_data(dot_tree_data.getvalue())
    dtree_graph.write_png(fileName)

# Exercise 1 (10 marks)
# Create a CART decision tree with splitting criterion as entropy
# Remember to set the random state to 1234

# Instantiate the classifier with appropriate parameters
dt = DecisionTreeClassifier(criterion='entropy', random_state=1234)
# Train/fit the classifier with training data and correct labels
dt.fit(binary_class_training_data, binary_class_training_labels)
# Test the classifier with unseen data
dt_pred = dt.predict(binary_class_testing_data)
# Evaluate the performance of classifier
score = dt.score(binary_class_testing_data, binary_class_testing_labels)
print('Decision tree score in exercise1: %.3f' %score)
plot_dtree(dt, 'dtree_pic.png')

nb = MultinomialNB()
nb.fit(binary_class_training_data, binary_class_training_labels)
nb_pred = nb.predict(binary_class_testing_data)
score = nb.score(binary_class_testing_data, binary_class_testing_labels)
print('MNB score in task2: %.3f' %score)

lr = LogisticRegression(random_state=1234)
lr.fit(binary_class_training_data, binary_class_training_labels)
lr_pred = lr.predict(binary_class_testing_data)
score = lr.score(binary_class_testing_data, binary_class_testing_labels)
print('LR score in task3: %.3f' %score)

rf = RandomForestClassifier(random_state=1234)
rf.fit(binary_class_training_data, binary_class_training_labels)
rf_pred = rf.predict(binary_class_testing_data)
score = rf.score(binary_class_testing_data, binary_class_testing_labels)
print('RF score in task4:')
print(score)

clf_pred = [dt_pred, nb_pred, lr_pred, rf_pred]
clf = [dt, nb, lr, rf]
# part2
# task t5a (5 marks)
# Print the classification report and confusion matrix for each of the models above
def t5a(pred):
    report = metrics.classification_report(binary_class_testing_labels, pred)
    print(report)
    cof_matrix = metrics.confusion_matrix(binary_class_testing_labels, pred)
    print(cof_matrix)

for i in clf_pred:
    t5a(i)

#t5b and t5c
fig = plt.figure()
fig.set_figwidth(10)
fig.suptitle('AUC for Classifier Predicting')
def roc(clf, count):
    fpr_p, tpr_p, thresholds_p = metrics.roc_curve(binary_class_testing_labels, clf.predict_proba(binary_class_testing_data)[:,1], pos_label=9)
    ax1 = plt.subplot(2,2,count+1)
    ax1.set_xlabel('false positive rate')
    ax1.set_ylabel('true positive rate')
    ax1.plot(fpr_p, tpr_p)

    print("False-positive rate:", fpr_p)
    print("True-positive rate: ", tpr_p)
    print("Thresholds:         ", thresholds_p)
    print('auc0')
    print(metrics.auc(fpr_p,tpr_p))
    print('auc1',metrics.roc_auc_score(binary_class_testing_labels,clf.predict(binary_class_testing_data)))

for index, item in enumerate(clf):
    roc(item, index)
plt.show()


#t5d
fig2 = plt.figure()
fig2.suptitle("Presion_ recall curve")
def t5d(clf, count):
    precision, recall, thresholds = metrics.precision_recall_curve(binary_class_testing_labels,
                                                   clf.predict_proba(binary_class_testing_data)[:, 1], pos_label=9)
    plt.subplot(2, 2, count + 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.step(recall, precision)
    plt.fill_between(recall, precision)

for index, item in enumerate(clf):
    t5d(item, index)
plt.show()


#part3
###Do not make any change below
# set the cv parameter of GridSearchCV to 3.
# remember to set the verbose parameter to 2
all_scaled_data = binary_class_data / 255.0
all_scaled_target = binary_class_labels

# Exercise 6 (15 marks)
# Tuning Random Forest for MNIST
# tuned_parameters = [{'max_features': ['sqrt', 'log2'], 'n_estimators': [1000, 1500]}]
#
# # Write code here
# clf = GridSearchCV(RandomForestClassifier(), cv=3, verbose= 2, param_grid=tuned_parameters)
# clf.fit(all_scaled_data, all_scaled_target)
# ans0 = clf.cv_results_
# print(ans0)
# ans1 = clf.best_estimator_
# print(ans1)
# ans2 = clf.best_params_
# print(ans2)
# ans3 = clf.best_index_
# print(ans3)

best = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

best.fit(binary_class_training_data, binary_class_training_labels)
# rf_pred = rf.predict(binary_class_testing_data)
score = best.score(binary_class_testing_data, binary_class_testing_labels)
print('best RF score:')
print(score)


# print the details of the best model and its accuracy
# Write code here


