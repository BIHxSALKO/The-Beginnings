# -*- coding: utf-8 -*-
"""
@author: asalkanovic1
"""

# Load Libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']     # Column names found in the iris.names file
dataset = pd.read_csv(url, names=names)

'''
--Summarize the Dataset--

After loading the data set in, we now want to take a look at the data in the following ways:
    1. Dimensions of the dataset.
    2. Peek at the data itself.
    3. Statistical summary of all attributes.
    4. Breakdown of the data by the class variable.
'''

# Dimensinos of the Dataset: shape
print(dataset.shape)

# Peek at the data
print(dataset.head(100))

# Statistical Summary: descriptions
print(dataset.describe())

# Class Distribution: this is for the 'class' column in the dataset
print(dataset.groupby('class').size())

'''
--Data Visualizations--

   1. Univariate plots to better understand each attribute.
   2. Multivariate plots to better understand the relationships between attributes.
'''

# Univariate Plots

# Box and Whisker Plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey= False)
plt.show()

# Histograms
dataset.hist()
plt.show()

# Multivariate Plots

# Scatter Plot Matrix
scatter_matrix(dataset)
plt.show()

'''
--Evaluate Some Algorithms--

Now it is time to create some models of the data and estimate their accuracy on unseen data.
Here is what we are going to cover in this step:
    1. Separate out a validation dataset.
    2. Set-up the test harness to use 10-fold cross validation.
    3. Build 5 different models to predict species from flower measurements
    4. Select the best model.
'''

# Create a Validation Dataset
'''
We will split the loaded data set into two, 80% of which will use to train our models
and 20% that we will hold back as a validation dataset.
'''

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness
'''
We will use 10-fold cross validation to estimate accuracy.
This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
'''
seed = 7
scoring = 'accuracy'

# Build Models
'''
We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some 
of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.

Let’s evaluate 6 different algorithms:
    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM).
This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed 
before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the 
results are directly comparable.
'''
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# Evaluate ead model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

'''
--Make Predictions--

The KNN algorithm was the most accurate model that we tested [SINCE TUTORIAL IS DATED AND THE DATA HAS SINCE BEEN UPDATED, SVM IS BETTER MODEL]. 
Now we want to get an idea of the accuracy of the model on our validation set.

This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just 
in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly 
optimistic result.

We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix 
and a classification report.
'''
# Make Predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# JUST TO PROVE THAT THE SVM is more accurate model currently due to descrepancy of time release of tutorial and newly updated data source
svm = SVC()
svm.fit(X_train, Y_train)
predictions2 = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions2))
print(confusion_matrix(Y_validation, predictions2))
print(classification_report(Y_validation, predictions2))
