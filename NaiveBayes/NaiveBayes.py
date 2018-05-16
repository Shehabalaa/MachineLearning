#!/usr/bin/python
import sys
from time import time
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# here we get features and labels for train and test in tuple
features_train, features_test, labels_train, labels_test = preprocess()
clf = GaussianNB()
t0=time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0),"s"
t1=time()
pred = clf.predict(features_test)
print "predicting time:",round(time()-t1),"s"
print "accuracy:",round(accuracy_score(pred,labels_test)*100)


#Sara has label 0
#Chris has label 1
