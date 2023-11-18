#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
df =datasets.load_diabetes()
df['feature_names']
diabetes_X,diabetes_y=datasets.load_diabetes(return_X_y=True)
diabetes_X.shape
diabetes_y.shape
diabetes_X=diabetes_X[:,np.newaxis,2]
diabetes_X.shape
diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]
diabetes_y_train=diabetes_y[:-20]
diabetes_y_test=diabetes_y[-20:]
regr=linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_pred=regr.predict(diabetes_X_test)
print("coefficients :\n",regr.coef_)
print("meansquareerror:%2f"%mean_squared_error(diabetes_y_test,diabetes_y_pred))
print("coefficient of determination:%2f"%r2_score(diabetes_y_test,diabetes_y_pred))
plt.scatter(diabetes_X_test,diabetes_y_pred,color="blue",linewidth=3)
plt.ylabel("diabetes progression")
plt.xticks(())
plt.yticks(())
plt.show(())


output
coefficients :
 [938.23786125]
meansquareerror:2548.072399
coefficient of determination:0.472575




