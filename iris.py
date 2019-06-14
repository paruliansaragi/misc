from sklearn import datasets
import numpy as np 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()



X = iris["data"][:, (2,3)]
y = (iris["target"] == 2).astype(np.float64)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


svm_pipe = Pipeline([
	("scaler", StandardScaler()),
	("linear_svc", LinearSVC(C=1, loss="hinge"))
	])

svm_pipe.fit(X_train, y_train)
print(svm_pipe.predict([[5.5, 1.7]]))

svm_cross_val_score = cross_val_score(svm_pipe, X_train, y_train, cv=10)
print('SVM Classifier Cross Validation Score model Score: ')
print(svm_cross_val_score.mean())
print('Random Forest Classifier Score model Score: ')
print(svm_pipe.score(X_test, y_test))