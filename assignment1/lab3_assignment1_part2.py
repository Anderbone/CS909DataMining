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
from sklearn.svm import SVC, LinearSVC , SVR
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
# mnist = fetch_mldata("MNIST original")

# In order to make the experiments repeatable, we will seed the random number generator to a known value
# That way the results of the experiments will always be same
np.random.seed(1234)

# Recall that we previously used shuffle and assigned first 5000 data as training and remaining as testing
# Here is another way to do this
# Here we are using a function in cross validation module to split
# By convention, let us use a 70/30 split
train_data, test_data, train_labels, test_labels = \
    train_test_split(mnist.data, mnist.target, test_size=0.3)

# The images are in grey scale where each number is between 0 to 255
# Now let us normalize them so that the values are between 0 and 1.
# This will be the only modification we will make to the image
train_data = train_data / 255.0
test_data = test_data / 255.0

# Plot the average value of all digits
plt.figure()
fig, axes = plt.subplots(2, 5, figsize=(15, 4))

for i in range(10):
    row, col = i // 5, i % 5

    # Change below: Subset p3_train_data with images for digit i only
    # Possible to do it 1 liner (similar to how it is done in Pandas)
    digit_i_subset = train_data[train_labels == i]

    # Change below: compute avg value of t3a_digit_i_subset
    # remember to use a vectorized version of mean for efficiency
    digit_i_subset_mean = np.mean(digit_i_subset, axis=0)

    # Do not change below
    axes[row][col].imshow(digit_i_subset_mean.reshape(28, 28), cmap="Greys")
    axes[row][col].grid(False)
    axes[row][col].get_xaxis().set_ticks([])
    axes[row][col].get_yaxis().set_ticks([])

# Task t7a (5 marks)
# Train a multi-class classifier (OneVsRest) with LinearSVC class and make predictions
# Print the training time and classification accuracy on the test set
# Write your code here
# svc = LinearSVC(random_state=0, tol=1e-5)
svc = LinearSVC(multi_class='ovr', random_state=1234)
start = time.time()
svc.fit(train_data, train_labels)
score = svc.score(test_data, test_labels)
print(score)
print('training time in t7a: ' + str(time.time() - start))



# Task t7b (10 marks)
# Perform PCA on the training data and map both training and test data into 100-dimensional space by PCA,
# Train a multi-class classifier (OneVsRest) with LinearSVC class using the trainformed training data,
# Print the training time and classification accuracy on the test set
# Remember that MNIST images are 28x28 => 784 features.

# You might want to check http://scikit-learn.org/stable/modules/decomposition.html#decompositions for details
# Write your code below
pca = PCA(n_components=100)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)

print("original shape:   ", train_data.shape)
print("transformed shape:", train_data_pca.shape)

clf = LinearSVC(multi_class='ovr')
start = time.time()
svc.fit(train_data_pca, train_labels)
score = svc.score(test_data_pca, test_labels)
print(score)
print('training time in t7b: ' + str(time.time() - start))


# Task t7c (10 marks)

# One way to determine how much components needs for PCA is to find the smallest value
# such that it explained 95% of the variance.
# Using the PCA results obtained above, print the cumulative variance that is explained
# by 100 components.
# Write your code below

# print(test_data_pca)
# print(len(test_data_pca))
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
print(var)
print(var[0])
print(var[99])

pca = PCA(n_components=500)

pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)
clf = LinearSVC(multi_class='ovr')
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
for i,item in enumerate(var):
    if float(var[i]) > 95.0:
        print(var[i])
        print(i)
        break
