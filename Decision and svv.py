PROGRAM DECISION TREE

import pandas as pd

import numpy as np

from sklearn.datasets import load_iris



#load iris data

data = load_iris()

data.data.shape

print('classes to predict: ',data.target_names)

print('Features: ',data.feature_names)

X = data.data

y = data.target



display (X.shape, y.shape)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 50, test_size = 0.25)

#default criterion is Gini

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy on train data using Gini: ',accuracy_score(y_true = y_train, y_pred = classifier.predict(X_train)))

print('Accuracy on test data using Gini: ',accuracy_score(y_true = y_test, y_pred = y_pred))

#change criterion to entropy

classifier_entropy = DecisionTreeClassifier(criterion='entropy')

classifier_entropy.fit(X_train, y_train)

y_pred_entropy = classifier_entropy.predict(X_test)

print('Accuracy on train data using entropy', accuracy_score(y_true=y_train, y_pred = classifier_entropy.predict(X_train)))

print('Accuracy on test data using entropy', accuracy_score(y_true=y_test, y_pred = y_pred_entropy))

#change criterion to entropy with min_samples_split to 50. Default value is 2

classifier_entropy1 = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)

classifier_entropy1.fit(X_train, y_train)

y_pred_entropy1 = classifier_entropy1.predict(X_test)

print('Accuracy on train data using entropy', accuracy_score(y_true=y_train, y_pred = classifier_entropy1.predict(X_train)))

print('Accuracy on test data using entropy', accuracy_score(y_true=y_test, y_pred = y_pred_entropy1))

#visualise the decision tree



from sklearn.tree import export_graphviz

from six import StringIO

from IPython.display import Image

import pydotplus



dot_data = StringIO()

#the students can try using classifier, classifier_entropy and classifier_entropy1

#as first parameter below.

export_graphviz(classifier, out_file = dot_data,filled = True, rounded = True,special_characters = True, feature_names = data.feature_names, class_names = data.target_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())




OUTPUT

classes to predict:  ['setosa' 'versicolor' 'virginica']
Features:  ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
(150, 4)
(150,)
Accuracy on train data using Gini:  1.0
Accuracy on test data using Gini:  0.9473684210526315
Accuracy on train data using entropy 1.0
Accuracy on test data using entropy 0.9473684210526315
Accuracy on train data using entropy 0.9642857142857143
Accuracy on test data using entropy 0.9473684210526315
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[7], line 83
     79 from six import StringIO
     81 from IPython.display import Image
---> 83 import pydotplus
     87 dot_data = StringIO()
     89 #the students can try using classifier, classifier_entropy and classifier_entropy1
     90 
     91 #as first parameter below.

ModuleNotFoundError: No module named 'pydotplus'





















PROGRAM SVV


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
cancer=datasets.load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)
clf=svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("actual values",y_test)
print("predicted values",y_pred)
print("accuracy",metrics.accuracy_score(y_test,y_pred))
print("precision",metrics.precision_score(y_test,y_pred))
print("recall",metrics.recall_score(y_test,y_pred))



OUTPUT

actual values [1 1 0 0 1 0 1 1 1 0 0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 1 0 1 1 1 1 1
 0 1 1 0 1 0 1 1 1 0 1 0 0 1 1 0 1 1 0 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 1 1 1
 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 1 0 1 1
 0 1 1 0 0 0 1 0 1 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 1 1 0 1 1 0 0 1 0
 0 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1]
predicted values [1 1 0 0 1 0 1 1 1 0 0 0 1 0 0 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 1 0 1 1 1 1 1
 0 1 1 0 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 1 1 1
 0 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 1 0 0 1 0 0 1 1 0 1 0 1 1
 0 1 1 0 0 0 1 0 1 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 1 0
 0 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1]
accuracy 0.9649122807017544
precision 0.9811320754716981
recall 0.9629629629629629



