# Titanic-probel-kaggle-SVM-solution

# Importing the libraries
import numpy as np
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values
print(X)


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:])
X[:, 2:] = imputer.transform(X[:,2:])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print ("encoded data")
print(X)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [3,6]] = sc.fit_transform(X[:, [3,6]])
print("scaled data")
print(X)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)


#Importing Test Set

# Importing the dataset
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8] ].values

traveller_id = dataset2.iloc[:,0 ].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:, 2:])
X_test[:, 2:] = imputer.transform(X_test[:,2:])
print(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test[:,[2,5]] = sc.fit_transform(X_test[:,[2,5]])

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
print ("encoded data test")
print(X_test)

# %%[]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
survivor_pred = (np.concatenate((traveller_id.reshape(len(traveller_id),1), y_pred.reshape(len(y_pred),1)),1))
print(survivor_pred)

# printing the csv results
survivor_pred2 = pd.DataFrame(data=survivor_pred, columns=('PassengerId','Survived')) 
survivor_pred2.to_csv('final3.csv',index=False)
