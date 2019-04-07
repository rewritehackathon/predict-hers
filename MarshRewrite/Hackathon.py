#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB

x = []
y = []

with open('C:/Hackathon/Code/data.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ' ')
	for row in reader:
		x.append(row[0: (len(row))])

for i in x:
	i[0] = i[0].split(',')
	y.append(i[0][-1])
	del i[0][-1]

X = []
for i in x:
	X.append(i[0])
Y = []
for i in y:
	Y.append(i)

#print(str(x[0]) + "\n")
#print(str(x[0])  + "     " + str(y[4000]) + "\n")

#X = np.asarray(X)
#Y = np.asarray(Y)

x = []
y = []

for i in X:
	temp = []
	for j in i:
		temp.append(float(j))
	x.append(temp)

for i in Y:
	temp = []
	for j in i:
		temp.append(float(j))
	y.append(temp)

#print(y[0])

x = np.asarray(x)
y = np.asarray(y)
#print(x[0])

#Naive Bayes Classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

clfnb = GaussianNB()
clfnb.fit(x_train, y_train)

print("Naive Bayes classifier")
print(clfnb.score(x_test, y_test))
print("\n")


# In[27]:


import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import random

x = []
y = []

with open('C:/Hackathon/Code/data.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ' ')
	for row in reader:
		x.append(row[0: (len(row))])

for i in x:
	i[0] = i[0].split(',')
	y.append(i[0][-1])
	del i[0][-1]

X = []
for i in x:
	X.append(i[0])
Y = []
for i in y:
	Y.append(i)

#print(str(X[0]) + "\n")
#print(str(X[0])  + "     " + str(Y[4000]) + "\n")

X = np.asarray(X)
Y = np.asarray(Y)

for i in X:
	for j in i:
		j = float(j)

for i in Y:
	for j in i:
		j = float(j)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

#SVM classifier 
clf = svm.SVC()
clf.fit(x_train, y_train)

print("SVM rbf kernel Classifier")
print(clf.score(x_test, y_test))
print("\n")

#Decision Tree Classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clftree = tree.DecisionTreeClassifier()
clftree.fit(x_train, y_train)

print("Decision Tree Classifier")
print(clftree.score(x_test, y_test))
print("\n")

x = []
y = []

'''
for i in range(0, len(X)):
	a = random.uniform(0, 5)
	if a <= 1:
		x.append(X[i])
		y.append(Y[i])
print('random sampling')

#SVM polynomial classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)

clf = svm.SVC(kernel = 'poly')
clf.fit(x_train, y_train)

print("SVM polynomial kernel Classifier")
print(clf.score(x_test, y_test))
print("\n")'''


# In[28]:


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

x = []
y = []

with open('C:/Hackathon/Code/data.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ' ')
	for row in reader:
		x.append(row[0: (len(row))])

for i in x:
	i[0] = i[0].split(',')
	y.append(i[0][-1])
	del i[0][-1]

X = []
for i in x:
	X.append(i[0])
Y = []
for i in y:
	Y.append(i)

#print(str(X[0]) + "\n")
#print(str(X[0])  + "     " + str(Y[4000]) + "\n")

X = np.asarray(X)
Y = np.asarray(Y)

#Random Forest Classifier
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

print("Random Forest classifier")
print(clf.score(x_test, y_test))
print("\n")

#Adaboost Classifier
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = AdaBoostClassifier()
clf.fit(x_train, y_train)

print("AdaBoost classifier")
print(clf.score(x_test, y_test))
print("\n")

#BaggingClassifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = BaggingClassifier()
clf.fit(x_train, y_train)

print("Bagging classifier")
print(clf.score(x_test, y_test))
print("\n")

#ExtraTreesClassifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = ExtraTreesClassifier()
clf.fit(x_train, y_train)

print("ExtraTrees classifier")
print(clf.score(x_test, y_test))
print("\n")

#GradientBoostingClassifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = GradientBoostingClassifier()
clf.fit(x_train, y_train)

print("GradientBoostingClassifier")
print(clf.score(x_test, y_test))
print("\n")

#Just Something

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(x_train, y_train)

print("Just trying something")
print(bagging.score(x_test, y_test))
print("\n")

#KneighboursClassifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

print("KNeighborsClassifier")
print(clf.score(x_test, y_test))
print("\n")


# In[30]:


import numpy as np
import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

x = []
y = []

with open('C:/Hackathon/Code/data.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ' ')
	for row in reader:
		x.append(row[0: (len(row))])

for i in x:
	i[0] = i[0].split(',')
	y.append(i[0][-1])
	del i[0][-1]

X = []
for i in x:
	X.append(i[0])
Y = []
for i in y:
	Y.append(i)

#print(str(X[0]) + "\n")
#print(str(X[0])  + "     " + str(Y[4000]) + "\n")

X = np.asarray(X)
Y = np.asarray(Y)

x = []
y = []

for i in X:
	temp = []
	for j in i:
		temp.append(float(j))
	x.append(temp)

for i in Y:
	temp = []
	for j in i:
		temp.append(float(j))
	y.append(temp)

#print(y[0])

x = np.asarray(x)
y = np.asarray(y)

#Logistic Regression l1 classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

clfl1 = LogisticRegression(penalty = 'l1')
clfl1.fit(x_train, y_train)

print("Logistic Regression l1 type classifier")
print(clfl1.score(x_test, y_test))
print("\n")

#Logistic Regression l2 classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

clfl2 = LogisticRegression(penalty = 'l2')
clfl2.fit(x_train, y_train)

print("Logistic Regression l2 type classifier")
print(clfl2.score(x_test, y_test))
print("\n")


# In[32]:


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier

x = []
y = []

with open('C:/Hackathon/Code/data.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ' ')
	for row in reader:
		x.append(row[0: (len(row))])

for i in x:
	i[0] = i[0].split(',')
	y.append(i[0][-1])
	del i[0][-1]

X = []
for i in x:
	X.append(i[0])
Y = []
for i in y:
	Y.append(i)

#print(str(X[0]) + "\n")
#print(str(X[0])  + "     " + str(Y[4000]) + "\n")

X = np.asarray(X)
Y = np.asarray(Y)

x = []
y = []

for i in X:
	temp = []
	for j in i:
		temp.append(float(j))
	x.append(temp)

for i in Y:
	temp = []
	for j in i:
		temp.append(float(j))
	y.append(temp)

print(y[0])

x = np.asarray(x)
y = np.asarray(y)

#SGDClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

clf = SGDClassifier()
clf.fit(x_train, y_train)

print("SGDClassifier")
print(clf.score(x_test, y_test))
print("\n")

