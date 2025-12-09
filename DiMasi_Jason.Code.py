#!/usr/bin/env python
# coding: utf-8

# In[72]:


from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np


# In[73]:


def sklearn_dataset_to_df(dataset):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = pd.Series(dataset.target)
    return df

iris_df = sklearn_dataset_to_df(load_iris())
digits_df = sklearn_dataset_to_df(load_digits())
cancer_df = sklearn_dataset_to_df(load_breast_cancer())
wine_df = sklearn_dataset_to_df(load_wine())


# In[74]:


wine_df


# In[102]:


#Algorithms: Random Forest, KNN, SVC, Naive Bayes, ADABoost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split


# In[84]:


def knn_tuning(df):
    knn_parameters = {'n_neighbors': [i for i in range(2,10)]}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, knn_parameters)
    clf.fit(df.loc[:,df.columns != 'target'], df['target'])

    return clf.best_params_

def random_forest_tuning(df):
    rf_parameters = {'n_estimators': [20,50,100,120,150]}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_parameters)
    clf.fit(df.loc[:,df.columns != 'target'], df['target'])

    return clf.best_params_

def svc_tuning(df):
    svc_parameters = {'C': [1, 10, 100], 'gamma': [0.01, 0.1]}
    svc = SVC()
    clf = GridSearchCV(svc, svc_parameters)
    clf.fit(df.loc[:,df.columns != 'target'], df['target'])

    return clf.best_params_

def ada_tuning(df):
    ada_parameters = {'n_estimators': [20, 50, 100], 'learning_rate': [0.1, 1]}
    ada = AdaBoostClassifier()
    clf = GridSearchCV(ada, ada_parameters)
    clf.fit(df.loc[:,df.columns != 'target'], df['target'])

    return clf.best_params_


# In[106]:


n_iterations = 10
datasets = [iris_df, digits_df, cancer_df, wine_df]
dataset_names = ['iris', 'digits', 'cancer', 'wine']

print('Best Accuracy Score For Each Classifier on Each Dataset')
print('---------------------------------------------------------')
for i in range(len(datasets)):
    dataset = datasets[i]
    print(f"Dataset: {dataset_names[i]} ({np.shape(dataset)[0]} Samples, {np.shape(dataset)[1]-1} Parameters, {len(dataset['target'].value_counts())} Classes)")
    
    knn_params = knn_tuning(dataset)
    rf_params = random_forest_tuning(dataset)
    svc_params = svc_tuning(dataset)
    ada_params = ada_tuning(dataset)

    knn_scores = []
    rf_scores = []
    svc_scores = []
    gnb_scores = []
    ada_scores = []
    
    for i in range(n_iterations):
        X = dataset.loc[:,dataset.columns != 'target']
        y = dataset['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        knn = KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'])
        knn.fit(X_train, y_train)
        knn_scores += [knn.score(X_test, y_test)]

        rf = RandomForestClassifier(n_estimators=rf_params['n_estimators'])
        rf.fit(X_train, y_train)
        rf_scores += [rf.score(X_test, y_test)]

        svc = SVC(C=svc_params['C'], gamma=svc_params['gamma'])
        svc.fit(X_train, y_train)
        svc_scores += [svc.score(X_test, y_test)]

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb_scores += [gnb.score(X_test, y_test)]

        ada = AdaBoostClassifier(n_estimators=ada_params['n_estimators'], learning_rate=ada_params['learning_rate'])
        ada.fit(X_train, y_train)
        ada_scores += [ada.score(X_test, y_test)]

    print(f"Best KNN Score: {np.max(knn_scores):.6f} | Mean KNN Score: {np.mean(knn_scores):.6f}")
    print(f"Best Random Forest Score: {np.max(rf_scores):.6f} | Mean Random Forest Score: {np.mean(rf_scores):.6f}")
    print(f"Best SVC Score: {np.max(svc_scores):.6f} | Mean SVC Score: {np.mean(svc_scores):.6f}")
    print(f"Best Naive Bayes Score: {np.max(gnb_scores):.6f} | Mean Naive Bayes Score: {np.mean(gnb_scores):.6f}")
    print(f"Best AdaBoost Score: {np.max(ada_scores):.6f} | Mean AdaBoost Score: {np.mean(ada_scores):.6f}")
    print('-----------------------------------------------')


# In[ ]:




