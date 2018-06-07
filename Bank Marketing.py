# # Introduction
# # 
# # Welcome to the **Cloudera Data Science Workbench Workshop**!
# # In this workshop, we'll walk through the core data science process using Python and CDSW.
# # Throughout this workshop, we will walk through the following subjects:
# # 
# # * Data Ingestion
# # * Data Preparation
# # * Analytical and Predictive Modeling
# 
# # By the end of this workshop, you should be comfortable using Python to perform basic data science tasks in CDSW.  If you would like to expand your skills further, Cloudera provides a number of interesting projects, complete with code, at this location:
# # http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#
# # You can find this demo at https://github.com/BreakingBI/CDSW-Demos
# 
# 
# # Agenda
# 
# Ingestion
# Exploration and Vizualization
# Feature Correlation
# Missing Values
# Dummy Coding
# Writing Files
# Model Training
# Model Evaluation
# 
#
# # Context
# 
# 
# age (numeric)
# job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# education (categorical: basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# default: has credit in default? (categorical: 'no','yes','unknown')
# housing_loan: has housing loan? (categorical: 'no','yes','unknown')
# credit: credit score (numeric)
# auto_loan: will apply for housing loan? (categorical: 'no','yes')

# # Workshop

# Import the Dataset

import pandas as pd

df = pd.read_csv('bank-additional-full.csv', low_memory=False)

!pip3 install s3fs
import s3fs

df = pd.read_csv('s3://awstestdrive/bank-additional-full.csv')

# Make Modifications

import numpy as np
import math
df = df[['age', 'job', 'marital', 'education', 'default', 'housing','y']]
df = df.rename(columns={'housing':'housing_loan', 'y':'auto_loan'})
df.marital = df.marital.replace('unknown', np.NaN)
df.loc[df.sample(math.ceil(df.marital.size / 10)).index, ['marital']] = np.NaN
df['credit'] = 1
df.loc[df.auto_loan == 'yes', ['credit']] = 700 + np.random.randint(low=-150, high=150, size=df.credit.size)
df.loc[df.auto_loan == 'no', ['credit']] = 600 + np.random.randint(low=-150, high=150, size=df.credit.size)

# Single Variable Visualizations

import seaborn as sns
sns.countplot(y='auto_loan', data=df)

_ = sns.countplot(y='auto_loan', data=df)

_ = sns.countplot(y='marital', data=df)
_ = sns.countplot(y='housing_loan', data=df)

df.age

df.age.hist()

_ = sns.boxplot(x='age', data=df)

# Multi-Variable Visulations, aka Feature Correlations

_ = sns.boxplot(x='age', y ='auto_loan', data=df)

_ = sns.boxplot(x='age', y='auto_loan', hue='marital', data=df)

_ = sns.boxplot(x='age', y='marital', hue='auto_loan', data=df)

_ = sns.countplot(y='education', hue='auto_loan', data=df)

df.education.value_counts()

df.education.value_counts().index

_ = sns.countplot(y='education', hue='auto_loan', data=df, order=df.education.value_counts().index)

_ = sns.regplot(x='age', y='credit', data=df)

#This command takes a second to run
_ = sns.pairplot(hue='auto_loan', data=df, kind='reg')

df.corr()

_ = sns.heatmap(df.corr())

# #Handling Missing Values

df.isnull()

df.isnull().sum()

df.isnull().sum() / df.age.size

df.marital.value_counts()

df.marital.fillna(value='unknown')

df.marital.fillna(value='unknown').value_counts()

df.marital.mode()

df.marital.mode()[0]

df.marital.fillna(value=df.marital.mode()[0]).valuecount(*)

df.marital.fillna(value='unknown', inplace=True)

df.marital.value_counts()
df.isnull().sum()

# Dummy Coding

df.job.value_counts()

pd.get_dummies(df.job)

job_dummy = pd.get_dummies(df.job)
job_dummy.columns

['job_' + col for col in job_dummy.columns]

job_dummy.columns = ['job_' + col for col in job_dummy.columns]

job_dummy.columns.str.replace(pat='-', repl='_')

job_dummy.columns = job_dummy.columns.str.replace(pat='-', repl='_')
job_dummy.columns = job_dummy.columns.str.replace(pat='.', repl='_')
job_dummy.columns

marital_dummy = pd.get_dummies(df.marital)
marital_dummy.columns = ['marital_' + col for col in marital_dummy.columns]
education_dummy = pd.get_dummies(df.education)
education_dummy.columns = ['education_' + col for col in education_dummy.columns]
education_dummy.columns = education_dummy.columns.str.replace(pat='.', repl='_')
default_dummy = pd.get_dummies(df.default)
default_dummy.columns = ['default_' + col for col in default_dummy.columns]
housing_loan_dummy = pd.get_dummies(df.housing_loan)
housing_loan_dummy.columns = ['housing_loan_' + col for col in housing_loan_dummy.columns]

pd.get_dummies(df.auto_loan).columns

pd.get_dummies(df.auto_loan).yes

auto_loan_dummy = pd.get_dummies(df.auto_loan)
auto_loan_dummy.columns

auto_loan_dummy.drop(['no'], axis=1, inplace=True)
auto_loan_dummy.columns

auto_loan_dummy.columns = ['auto_loan_yes']
auto_loan_dummy.columns

pd.concat([default_dummy, housing_loan_dummy], axis=1)

df_dummy = pd.concat([df, job_dummy, marital_dummy, education_dummy, default_dummy, housing_loan_dummy, auto_loan_dummy], axis=1)

df_dummy.drop(['job', 'marital', 'education', 'default', 'housing_loan', 'auto_loan'], axis=1, inplace=True)
df_dummy.columns

# Write the finalized file back to storage

df_dummy.head()

df_dummy.to_csv('encoded_bank_marketing_dummy.csv')

df_dummy.to_csv('encoded_bank_marketing_dummy.tsv', sep='\t')

# Predictive Model Training

X = df_dummy.drop('auto_loan_yes', axis=1)
X.shape

y = df_dummy.auto_loan_yes
y.shape

!pip3 install sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn import tree
dtree_clf = tree.DecisionTreeClassifier(max_leaf_nodes=3)

_ = dtree_clf.fit(X_train, y_train)

# Install from terminal:
# conda install graphviz
# pip3 install graphviz

import graphviz
dtree_export = tree.export_graphviz(dtree_clf, out_file=None) 
dtree_graph = graphviz.Source(dtree_export) 
dtree_graph

dtree_clf = tree.DecisionTreeClassifier()
_ = dtree_clf.fit(X_train, y_train)

dtree_clf.predict(X_test)

dtree_pred = dtree_clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, dtree_pred)

dtree_acc = accuracy_score(y_test, dtree_pred)

1 - accuracy_score(y_test, dtree_pred)

from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier()
_ = rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

dtree_acc
rf_acc

gb_clf = ensemble.GradientBoostingClassifier()
_ = gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

dtree_acc
rf_acc
gb_acc