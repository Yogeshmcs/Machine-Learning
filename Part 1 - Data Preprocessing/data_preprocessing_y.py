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
x[:,0]=labelencoder_x.fit_transform(x[:,0]) # it just assign values which is not good as ML algo will consider these as weights
onehotencoder = OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray() # it creates many column (2^n) and assign binary sequence hence no problem of condidering as weight each new colmns corresponds to it's respective category.
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y) #here no probles as here ony two categoris



# Spliting the data into test_set and training_set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#