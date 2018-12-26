# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:16:53 2018

@author: Anuj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Training set
df = pd.read_csv('Train.csv')
#Test set
df_test = pd.read_csv('Test.csv')

#Summary of dataset
print(df.describe())

#Checking for null values in dataset
print(df.isnull().sum())

#Checking the different entries in each column with the count
print(df.Item_Fat_Content.value_counts())

print(df.Item_Type.value_counts())

print(df.Outlet_Establishment_Year.value_counts())

print(df.Outlet_Size.value_counts())

print(df.Outlet_Type.value_counts())

#Relation between Outlet_Size and Outlet_Type
#plt.scatter(df['Outlet_Size'], df['Outlet_Type'])
#plt.show()

#Above graph shows that grocery store is always small. Hence, map grocery store to small if grocery store is nan
d = {'Grocery Store' : 'Small'}
s = df.Outlet_Type.map(d)
df.Outlet_Size = df.Outlet_Size.combine_first(s)

df.Outlet_Size.isnull().any()

#Relation between Outlet_Size and Outlet_Location_Type
#plt.scatter(df['Outlet_Size'],df['Outlet_Location_Type'])
#plt.show()

d={'Tier 2':'Small'}
s=df.Outlet_Location_Type.map(d)
df.Outlet_Size=df.Outlet_Size.combine_first(s)

df.Outlet_Size.isnull().any()

#Fill missing items in columns by grouping and calculating mean
 
df['Item_Weight']=df['Item_Weight'].fillna(df.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
print(df.Item_Weight.isnull().sum())

List=['Baking Goods','Breads','Breakfast','Canned','Dairy','Frozen Foods','Fruits and Vegetables','Hard Drinks','Health and Hygiene','Household','Meat','Others','Seafood','Snack Foods','Soft Drinks','Starchy Foods']
Mean_values_Item_Type_data=df.groupby('Item_Type')['Item_Weight'].mean()

for i in List:
    d={i:Mean_values_Item_Type_data[i]}
    s=df.Item_Type.map(d)
    df.Item_Weight=df.Item_Weight.combine_first(s)
Mean_values_Item_Type_data=df.groupby('Item_Type')['Item_Weight'].mean()

print(df.Item_Visibility.value_counts())
df['Item_Visibility'] = df['Item_Visibility'].replace(0.000000, np.nan)
df['Item_Visibility'] = df['Item_Visibility'].fillna(df.groupby('Item_Fat_Content')['Item_Visibility'].transform('mean')) 
print(df.Item_Visibility.isnull().any())

#Converting irregular entries into uniform entries
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['reg'], 'Regular')
print(df.Item_Fat_Content.unique())

#Encode categorical data using pd.get_dummies
df = pd.get_dummies(df, columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], drop_first = True)
df.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1, inplace = True)

from scipy.stats import skew

Y = df['Item_Outlet_Sales']
X = df.drop(['Item_Outlet_Sales'], axis = 1)

#Check for skewness
s = skew(Y)

#Apply log to remove skew
Y = np.log1p(Y)

#Split dataset into train and validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

cols = ['Item_Weight', 'Item_MRP', 'Item_Visibility', 'Outlet_Establishment_Year']

for i in range(len(cols)):
    fig, ax = plt.subplots()
    ax.scatter(x = df[cols[i]], y = df['Item_Outlet_Sales'])
    plt.ylabel('Sales', fontsize = 13)
    plt.xlabel(cols[i], fontsize = 13)
    plt.show()

#Fitting the correct model
    
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import KFold, cross_val_score
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#Ridge regression    
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha = 0.08, normalize = True)
score = rmsle_cv(ridgeReg)
print('Score for Ridge: {:.4f}'.format(score.mean()))

#Random forest and Gradient Boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rfreg = RandomForestRegressor(n_estimators = 100, criterion = 'mse')
score = rmsle_cv(rfreg)
print('Score for Random Forest: {:.4f}'.format(score.mean()))

GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print('Score for GBoost: {:.4f}'.format(score.mean()))

#ElasticNet and Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("Score for ElasticNet {:.4f}".format(score.mean()))

lassoreg = make_pipeline(RobustScaler(), Lasso(alpha = 0.03, normalize = True))
score = rmsle_cv(lassoreg)
print('Score for Lasso: {:.4f}'.format(score.mean()))

#Fit the models for training set
ridgeReg.fit(X_train, y_train)
lassoreg.fit(X_train, y_train)
rfreg.fit(X_train, y_train)
GBoost.fit(X_train, y_train)
ENet.fit(X_train, y_train)

#Predict values for training set
ridge_pred_train = np.expm1(ridgeReg.predict(X_train))
lasso_pred_train = np.expm1(lassoreg.predict(X_train))
rfreg_pred_train = np.expm1(rfreg.predict(X_train))
GBoost_pred_train = np.expm1(GBoost.predict(X_train))
ENet_pred_train = np.expm1(ENet.predict(X_train))

#Check MSE for different models
y_train = np.expm1(y_train)
rms_ridge = sqrt(mean_squared_error(y_train, ridge_pred_train))
print(" ")
print("MSE for Ridge:", rms_ridge)
rms_rfreg = sqrt(mean_squared_error(y_train, rfreg_pred_train))
print("MSE for Random Forest:",rms_rfreg)
rms_gboost = sqrt(mean_squared_error(y_train, GBoost_pred_train))
print("MSE for GBoost:",rms_gboost)
rms_enet = sqrt(mean_squared_error(y_train, ENet_pred_train))
print("MSE for ElasticNet",rms_enet)
rms_lasso = sqrt(mean_squared_error(y_train, lasso_pred_train))
print("MSE for Lasso:",rms_lasso)

#RMS of the model for training set
Final_train = ((rfreg_pred_train) + (GBoost_pred_train))/2
rms_train = sqrt(mean_squared_error(y_train, Final_train))
print(" ")
print("RMS for training set:", rms_train)

#Predicting values for validation set
ridge_pred = ridgeReg.predict(X_test)
lasso_pred = lassoreg.predict(X_test)
rfreg_pred = rfreg.predict(X_test)
GBoost_pred = GBoost.predict(X_test)
ENet_pred = ENet.predict(X_test)

#RMS of the model for validation set
Final = (np.expm1(rfreg_pred) + np.expm1(GBoost_pred))/2
y_test = np.expm1(y_test)
rms_test = sqrt(mean_squared_error(y_test, Final))
print("RMS for validation set:", rms_test)

#TESTING
#Above graph shows that grocery store is always small. Hence, map grocery store to small if grocery store is nan
d = {'Grocery Store' : 'Small'}
s_test = df_test.Outlet_Type.map(d)
df_test.Outlet_Size = df_test.Outlet_Size.combine_first(s_test)

df_test.Outlet_Size.isnull().any()

#Relation between Outlet_Size and Outlet_Location_Type
#plt.scatter(df['Outlet_Size'],df['Outlet_Location_Type'])
#plt.show()

d={'Tier 2':'Small'}
s_test=df_test.Outlet_Location_Type.map(d)
df_test.Outlet_Size=df_test.Outlet_Size.combine_first(s_test)

df_test.Outlet_Size.isnull().any()

#Fill missing items in columns by grouping and calculating mean
 
df_test['Item_Weight']=df_test['Item_Weight'].fillna(df_test.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
print(df_test.Item_Weight.isnull().sum())

List=['Baking Goods','Breads','Breakfast','Canned','Dairy','Frozen Foods','Fruits and Vegetables','Hard Drinks','Health and Hygiene','Household','Meat','Others','Seafood','Snack Foods','Soft Drinks','Starchy Foods']
Mean_values_Item_Type_data=df_test.groupby('Item_Type')['Item_Weight'].mean()

for i in List:
    d={i:Mean_values_Item_Type_data[i]}
    s=df_test.Item_Type.map(d)
    df_test.Item_Weight=df_test.Item_Weight.combine_first(s)
Mean_values_Item_Type_data=df_test.groupby('Item_Type')['Item_Weight'].mean()

print(df_test.Item_Visibility.value_counts())
df_test['Item_Visibility'] = df_test['Item_Visibility'].replace(0.000000, np.nan)
df_test['Item_Visibility'] = df_test['Item_Visibility'].fillna(df_test.groupby('Item_Fat_Content')['Item_Visibility'].transform('mean')) 
print(df_test.Item_Visibility.isnull().sum())

#Converting irregular entries into uniform entries
df_test['Item_Fat_Content'] = df_test['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
df_test['Item_Fat_Content'] = df_test['Item_Fat_Content'].replace(['reg'], 'Regular')
print(df_test.Item_Fat_Content.unique())

#Encode categorical data using pd.get_dummies
df_test = pd.get_dummies(df_test, columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], drop_first = True)
df_test.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1, inplace = True)

#Predicting using selected models
random_forest_pred = rfreg.predict(df_test)
Gradient_Boost_pred = GBoost.predict(df_test) 

Prediction = (np.expm1(random_forest_pred) + np.expm1(Gradient_Boost_pred))/2