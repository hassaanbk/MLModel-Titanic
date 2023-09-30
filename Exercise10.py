#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:45:01 2022

@author: Hassaan
"""

import numpy as np
import pandas as pd
import os

path = "/Users/Hassaan/Desktop/Data Warehousing and Predictive Analytics/Exercise 10"
filename = 'titanic3.csv'
fullpath = os.path.join(path, filename)
data_hassaan_i = pd.read_csv(fullpath, sep=',')
print(data_hassaan_i)
print(data_hassaan_i.columns.values) 
print(data_hassaan_i.describe())
print(data_hassaan_i.dtypes)
print(data_hassaan_i.isnull())

pd.set_option('display.max_columns', 30)  # set the maximum width
# Load the dataset in a dataframe object
df = pd.read_csv(fullpath, sep=',')
# Explore the data check the column values
print(df.columns.values)
print(df.head())
print(df.info())
categories = []
for col, col_type in df.dtypes.iteritems():
    if col_type == 'O':
         categories.append(col)
    else:
         df[col].fillna(0, inplace=True)
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.info()
#check for null values
print(len(df) - df.count())  # Cabin , boat, home.dest have so many missing values

df.loc[:,('age','sex', 'embarked', 'survived')].dropna(axis=0,how='any',inplace=True) 
df.info() 

categoricals = []
for col, col_type in df.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)

print(categoricals)

df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())


from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['age'].describe())
print(scaled_df['sex_male'].describe())
print(scaled_df['sex_female'].describe())
print(scaled_df['embarked_C'].describe())
print(scaled_df['embarked_Q'].describe())
print(scaled_df['embarked_S'].describe())
print(scaled_df['survived'].describe())
print(scaled_df.dtypes)

from sklearn.linear_model import LogisticRegression
import numpy as np

dependent_variable = 'survived'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)


testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import joblib 
joblib.dump(lr, '/Users/Hassaan/Desktop/Data Warehousing and Predictive Analytics/Exercise 10/model_lr2.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/Hassaan/Desktop/Data Warehousing and Predictive Analytics/Exercise 10/model_columns.pkl')
print("Models columns dumped!")

