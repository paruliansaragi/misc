'''import pandas as pd 

train_data = pd.read_csv('./datasets/titanic/train.csv')
test_data = pd.read_csv('./datasets/titanic/test.csv')

#print(train_data.head())
#The data is already split into train and test but test has no labels

# ~~ pre-processing pipelines
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

#Pipeline for numerical categories
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Imputer

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ('imputer', Imputer(strategy="median"))
    ])

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

#print(cat_pipeline.fit_transform(train_data))
#join numerical and categorical encoders
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

# ~~ now we have our pipelines that transform our data lets train our classifier
#from sklearn.svm import SVC

#svm_clf = SVC()
#svm_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.fit_transform(test_data)
#y_pred = svm_clf.predict(X_test)

#use cross_validation to see our scores
from sklearn.model_selection import cross_val_score

#svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
#print(svm_scores.mean())  0.7365

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(forest_scores.mean())  #~~ 0.8115690614005221'''


'''
Frame the Problem ~~ what are we trying to achieve? 
'''
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#Load the Data
train_data = pd.read_csv('./datasets/titanic/train.csv')
test_data = pd.read_csv('./datasets/titanic/test.csv')

'''print(train_data.info())
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64  
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object''' 
#print(train_data.head()) 5x12
#print(train_data.describe()) 
'''print(train_data.keys()) Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')'''
'''print(test_data.keys()) Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')'''

#Filtering out Null values
'''print(pd.isnull(train_data).sum())
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64'''

'''print(pd.isnull(test_data).sum())
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64'''

train_data.drop(labels=["Name", "Cabin", "Ticket", "PassengerId"], axis=1, inplace=True )
'''print(pd.isnull(train_data).sum())
PassengerId      0
Survived         0
Pclass           0
Sex              0
Age            177
SibSp            0
Parch            0
Fare             0
Cabin          687
dtype: int64'''

test_data.drop(labels=["Name", "Cabin", "Ticket", "PassengerId"], axis=1, inplace=True)
'''print(pd.isnull(test_data).sum())
PassengerId      0
Pclass           0
Sex              0
Age             86
SibSp            0
Parch            0
Fare             1
Cabin          327
dtype: int64'''

train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)


'''print(pd.isnull(test_data).sum())
PassengerId    0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
Embarked       0
dtype: int64'''
'''print(pd.isnull(train_data).sum())
PassengerId    0
Survived       0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
Embarked       2
dtype: int64'''
train_data["Embarked"].fillna("S", inplace=True)
'''print(pd.isnull(train_data).sum())
PassengerId    0
Survived       0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
Embarked       0
dtype: int64'''



