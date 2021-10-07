# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:40:24 2021

@author: User
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X,y = datasets.load_iris(return_X_y=True)
X.shape
y.shape

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4, random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

clf=svm.SVC(kernel="linear",C=1).fit(X_train,y_train)
clf.score(X_test,y_test)

##computing cross-validated metrics
from sklearn.model_selection import cross_val_score

clf=svm.SVC(kernel="linear", C=1, random_state=42)
scores=cross_val_score(clf,X,y,cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(),scores.std()))

##the scoring parameter control the evaluation method of score
##https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
##When cv=integer then default ShuffleSplit method is Kfold or StratifiedFfold
## It is also possible to create customize ShuffleSplit object
scores=cross_val_score(clf,X,y,cv=5, scoring="f1_macro")

##use cross_calidate can return the score of matric
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring)
sorted(scores.keys())

scores['test_recall_macro']


##Kfold 
import numpy as np
from sklearn.model_selection import KFold
X=["a","b","c","d"]
kf=KFold(n_splits=2)

for train, test in kf.split(X):
    print("%s %s" % (train,test))

#leave one out(leave p out)
from sklearn.model_selection import LeaveOneOut
X=[1,2,3,4]
loo=LeaveOneOut(P=1)
for train, test in loo.split(X):
    print("%s %s" % (train,test))

##Leave one group
from sklearn.model_selection import LeaveOneGroupOut

X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))
    
##example by iris data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
kf=KFold(n_splits=8)

clf=svm.SVC(kernel="linear", C=1, random_state=42)
scores=cross_val_score(clf,X,y,cv=kf.split(X))
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(),scores.std()))
