#!/usr/bin/env python

"""
Code for the machine learning model

By Nick, Cyrus, and Clarence
"""

import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import Counter

# helper funcitons
def error_by_feature(dataframe, column_dict, col_name, model):
	"""
	ARGS:
		dataframe: the main dataframe
		column_dict: the feature we are trying to isolate (one of the dictionaries from above)
		model: a machine learning model
		col_name: string of column name

	RETURN:
		dict containing column value and prediction error

		Example output: {"Male": 0.1675, "Female": 0.09125}
	"""
	test_dataframes = []
	names = []
	test_answers = []
	res = {}

	for item in column_dict:
		test_dataframes.append(dataframe[dataframe[col_name]==column_dict[item]])
		names.append(item)
	for dframe in test_dataframes:
		test_answers.append(dframe['income'])

	for i in range (len(test_dataframes)):
		test_dataframes[i] = test_dataframes[i].drop(columns="income")

	for i in range(len(names)):
		res[names[i]] = error(test_answers[i],optimal_dt.predict(test_dataframes[i]))

	return res



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

columns_with_mapping = {"workclass": workclass, "education": education, "marital_status": marital_status, "occupation": occupation, "relationship": relationship, "race": race, "sex": sex, "native_country": native_country, "income": income }



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


x_train = training_data.head(len(training_data) - 2000)
x_test = training_data.tail(2000)
y_train = income_answers.head(len(training_data) - 2000).ravel()
y_test = income_answers.tail(2000).ravel()

dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
Y_hat_dt = dt.predict(x_test)

dt_error_data = []
log_error_data = []
count=0


# create graph for error based on sample training size
for i in range(500,30000,500):
  dt.fit(x_train[:i], y_train[:i])
  dt_error_data.append(error(y_test, dt.predict(x_test)))
x_axis = range(500,30000,500)
x_label = "Sample Size"
y_label = "Test Error"
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.plot(x_axis, dt_error_data, label="DT error data")
plt.legend()
optimal_train_size = x_axis[dt_error_data.index(min(dt_error_data))]
print("Minimum error training size is: " + str(optimal_train_size))
plt.show()


# create graph for error vs d-tree depth
res_error = []
train_error = []
for i in range(1,16):
  dt = tree.DecisionTreeClassifier(max_depth=i)
  dt.fit(x_train,y_train)
  res_error.append(error(y_test,dt.predict(x_test)))
  train_error.append(error(y_train,dt.predict(x_train)))
x_axis_range = range(1,16)
x_label = "Max Depth"
y_label = "Test Error"
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.plot(x_axis_range,res_error, label="X Test error")
plt.plot(x_axis_range,train_error,label="X Train error")
plt.legend()
optimal_depth = x_axis_range[res_error.index(min(res_error))]
print("Minimum error for d-tree depth: " + str(optimal_depth))
plt.show()


# creating and training optimal model
optimal_dt = tree.DecisionTreeClassifier(max_depth=optimal_depth)
opt_x = training_data.head(optimal_train_size)
opt_y = income_answers.head(optimal_train_size).ravel()
optimal_dt.fit(opt_x, opt_y)


errors_by_feature = {}
for feature in columns_with_mapping:
	errors_by_feature[feature] = error_by_feature(dataframe, columns_with_mapping[feature], feature, optimal_dt)

length_by_feature = {}
for key in errors_by_feature:
	for feature in errors_by_feature[key]:
		if feature != ' ?':
			length_by_feature[feature] = len(dataframe[dataframe[key] == columns_with_mapping[key][feature]])


# graph that is errors by number of entries
x_axis = []
y_axis = []
while length_by_feature:
	feature = min(length_by_feature, key=length_by_feature.get)
	x_axis.append(min(length_by_feature.values()))
	for key in errors_by_feature:
		for x in errors_by_feature[key]:
			if x == feature:
				y_axis.append(errors_by_feature[key][x])
	del length_by_feature[feature]
x_label = "Number of Entries"
y_label = "Test Error"
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.plot(x_axis, y_axis, label="Error by Number of Entries")
plt.legend()
plt.show()

#determine if model is guessing if women and black people make less than 50k
#print predicitons for women and black people
black_data = training_data[training_data["race"]==4]
black_predicted = ((optimal_dt.predict(black_data)))
black_counts = Counter(black_predicted)


women_data = training_data[training_data["sex"]==0]
women_predicted = optimal_dt.predict(women_data)
women_counts = Counter(women_predicted)

men_data = training_data[training_data["sex"]==1]
men_predicted = optimal_dt.predict(men_data)
men_counts = Counter(men_predicted)

men_dt = dataframe[dataframe['sex']==1]
white_men = men_dt[men_dt['race']==0]
print(len(dataframe))
print(len(white_men))
