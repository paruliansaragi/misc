import pandas as pd 
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

#Load the Data
train_data = pd.read_csv('./datasets/titanic/train.csv')
test_data = pd.read_csv('./datasets/titanic/test.csv')
columns_train = ['Age', 'Pclass','Sex','Fare']
columns_target = ['Survived']
X = train_data[columns_train]
y = train_data[columns_target]

#Data Preprocessing

#Age            177
#Cabin          687
#Embarked         2

#Lets median the age missing values
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)

test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
#print(pd.isnull(train_data).sum())
test_data.drop(labels=["Name", "Cabin", "Ticket", "PassengerId", "Embarked"], axis=1, inplace=True)
train_data.drop(labels=["Name", "Cabin", "Ticket", "PassengerId", "Embarked"], axis=1, inplace=True)
#print(pd.isnull(train_data).sum())

#Lets encode gender 0 for male 1 for female
E = {'male': 0, 'female': 1}
X['Sex'] = X['Sex'].apply(lambda x:E[x])
#print(X['Sex'].head())
#print(pd.isnull(train_data).sum())
#print(train_data.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train["Age"].fillna(X_train["Age"].median(), inplace=True)
X_test["Age"].fillna(X_test["Age"].median(), inplace=True)
'''print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())'''

#preprocess test data
#X_test = test_data
#X_test['Sex'] = X_test['Sex'].apply(lambda x:E[x])
#print(X_test.head())


#y_train = train_data['Survived']
#X_train = train_data
#X_train.drop(labels=["Survived"], axis=1, inplace=True)
#y_train["Age"].fillna(y_train["Age"].median(), inplace=True)
#X_train["Age"].fillna(X_train["Age"].median(), inplace=True)

#np.isinf(y_train)
#print(pd.isnull(y_train).sum())
#print(np.isnan(y_train))
#print(X_train.head())
#lets train our model
#X_train['Sex'] = X_train['Sex'].apply(lambda x:E[x])
#print(X_train.head())
#print(y_train.head())
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
forest_scores = cross_val_score(rfc, X_train, y_train, cv=10)
print('Random Forest Classifier Cross Validation Score model Score: ')
print(forest_scores.mean())
print('Random Forest Classifier Score model Score: ')
print(rfc.score(X_test, y_test))

'''[0]
[0 0 1 1 0 0 1 0 1 1]
0.7457627118644068'''

from sklearn import svm
clf = svm.LinearSVC()
svm_scores = cross_val_score(clf, X_train, y_train, cv=10)
clf.fit(X_train, y_train)
print('Linear SVM Cross Validation Score model Score: ')
print(svm_scores.mean())
print('Linear SVM model Score: ')
print(clf.score(X_test, y_test))

from sklearn import svm
svm_clf = svm.SVC()

svm_clf.fit(X_train, y_train)
svm_clf_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print('SVM Cross Validation Score model Score: ')
print(svm_clf_scores.mean())

print('SVM model Score: ' )
print(svm_clf.score(X_test, y_test))