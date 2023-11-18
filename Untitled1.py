#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
iris=load_iris()
x=iris.data 
y=iris.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1) 
c_knn=KNeighborsClassifier(n_neighbors=3) 
c_knn.fit(x_train,y_train)
y_pred=c_knn.predict(x_test) 
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
sample=[[2,2,2,2]]
pred=c_knn.predict(sample) 
pred_v=[iris.target_names[p] for p in pred]
print(pred_v)

output
Accuracy: 0.9777777777777777
['setosa']






# In[6]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)
gnb=GaussianNB()
y_pred=gnb.fit(X_train,y_train).predict(X_test)
print(y_pred)
x_new=[[5,5,4,4]]
y_new=gnb.fit(X_train,y_train).predict(x_new)
print("predicted output for [[5,5,4,4]]:",y_new)
print("Naive bayes score:",gnb.score(X_test,y_test))

output
[2 1 0 2 0 2 0 1 1 1 1 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 1 1 1 2 0 2 0 0 1 2 2 1 2 1 2 1 1 2 1 1 2 1 2 1 0 2 1 1 1 1 2 0 0 2 1 0 0
 1]
predicted output for [[5,5,4,4]]: [2]
Naive bayes score: 0.9466666666666667





