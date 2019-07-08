# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:39:11 2019

@author: Yogesh Kushwah
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 3].values
print(x)

# Taking care of missing data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN',strategy = "mean", axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Spliting the data into test_set and training_set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#