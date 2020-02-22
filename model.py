#!/usr/bin/env python

"""
Code for the machine learning model

By Nick, Cyrus, and Clarence
"""

import pandas
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error
import numpy as np


def error(y, y_hat):
    diff = np.abs(y_hat-y)
    err = np.sum(diff)
    result = err/len(diff)
    return result

dataframe = pandas.read_csv("adult.data.csv")

"""
data in columns =
age: continuous.
workclass: Private
fnlwgt: continuous.
education: Bachelors
education-num: continuous.
marital-status: Married-civ-spouse
occupation: Tech-support
relationship: Wife
race: White
sex: Female
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States
"""

columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship",
 "race", "sex", "capital-gain", "capital_loss", "hours_per_week", "native_country", "income"]

dataframe.columns = columns

workclass = { " Private": 0, " Self-emp-not-inc": 1, " Self-emp-inc": 2, " Federal-gov": 3, " Local-gov": 4, " State-gov": 5, " Without-pay": 6, " ?": 7, " Never-worked": 8}
education = { " Bachelors": 0, " Some-college":1 ," 11th":2 ," HS-grad":3 ," Prof-school":4 ," Assoc-acdm":5 ," Assoc-voc":6 ," 9th":7 ," 7th-8th":8 ," 12th":9 ," Masters":10 ," 1st-4th":11 ," 10th":12 ," Doctorate":13 ," 5th-6th":14 ," Preschool": 15}
marital_status = { " Married-civ-spouse": 0, " Divorced": 1, " Never-married": 2, " Separated": 3, " Widowed": 4, " Married-spouse-absent": 5, " Married-AF-spouse": 6}
occupation = {" Tech-support": 0,  " Craft-repair": 1, " Other-service": 2, " Sales": 3, " Exec-managerial": 4, " Prof-specialty": 5, " Handlers-cleaners": 6, " Machine-op-inspct": 7, " Adm-clerical": 8, " Farming-fishing": 9, " Transport-moving": 10, " Priv-house-serv": 11, " Protective-serv": 12, " Armed-Forces": 13, " ?": 14}
relationship = {" Wife": 0, " Own-child": 1, " Husband": 2, " Not-in-family": 3, " Other-relative": 4, " Unmarried": 5}
race = {" White": 0, " Asian-Pac-Islander": 1, " Amer-Indian-Eskimo": 2, " Other": 3, " Black": 4}
sex = {" Female": 0, " Male": 1}
native_country = { " United-States": 1, " Cambodia": 2, " England": 3, " Puerto-Rico": 4, " Canada": 5, " Germany": 6, " Outlying-US(Guam-USVI-etc)": 7, " India": 8, " Japan": 9, " Greece": 10, " South": 11, " China": 12, " Cuba": 13, " Iran": 14, " Honduras": 15, " Philippines": 16, " Italy": 17, " Poland": 18, " Jamaica": 19, " Vietnam": 20, " Mexico": 21, " Portugal": 22, " Ireland": 23, " France": 24, " Dominican-Republic": 25, " Laos": 26, " Ecuador": 27, " Taiwan": 28, " Haiti": 29, " Columbia": 30, " Hungary": 31, " Guatemala": 32, " Nicaragua": 33, " Scotland": 34, " Thailand": 35, " Yugoslavia": 36, " El-Salvador": 37, " Trinadad&Tobago": 38, " Peru": 39, " Hong": 40, " Holand-Netherlands":41, " ?": 42}
income = {" >50K": 0, " <=50K": 1}

dataframe.workclass = [workclass[item] for item in dataframe.workclass]
dataframe.education = [education[item] for item in dataframe.education]
dataframe.marital_status = [marital_status[item] for item in dataframe.marital_status]
dataframe.occupation = [occupation[item] for item in dataframe.occupation]
dataframe.relationship = [relationship[item] for item in dataframe.relationship]
dataframe.race = [race[item] for item in dataframe.race]
dataframe.sex = [sex[item] for item in dataframe.sex]
dataframe.native_country = [native_country[item] for item in dataframe.native_country]
dataframe.income = [income[item] for item in dataframe.income]

income_answers = dataframe['income']

training_data = dataframe
training_data = training_data.drop(columns="income")

x_train = training_data.head(1000)
x_test = training_data.tail(1000)
y_train = income_answers.head(1000).ravel()
y_test = income_answers.tail(1000).ravel()

dt = tree.DecisionTreeClassifier(max_depth=5)
logreg = linear_model.LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train)
dt.fit(x_train, y_train)
Y_hat_logreg = logreg.predict(x_test)
Y_hat_dt = dt.predict(x_test)


print((error(y_test, Y_hat_dt), error(y_test, Y_hat_logreg)))













