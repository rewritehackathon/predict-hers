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

#Bagging Classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = BaggingClassifier()
clf.fit(x_train, y_train)

print("Bagging classifier")
print(clf.score(x_test, y_test))
print("\n")

#ExtraTrees Classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = ExtraTreesClassifier()
clf.fit(x_train, y_train)

print("ExtraTrees classifier")
print(clf.score(x_test, y_test))
print("\n")

#GradientBoosting Classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = GradientBoostingClassifier()
clf.fit(x_train, y_train)

print("GradientBoostingClassifier")
print(clf.score(x_test, y_test))
print("\n")

#Trying other classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(x_train, y_train)

print("Just trying something")
print(bagging.score(x_test, y_test))
print("\n")

#Kneighbours Classifier

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

print("KNeighborsClassifier")
print(clf.score(x_test, y_test))
print("\n")
