import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
from sklearn.cluster import KMeans
import seaborn as sns
import json
import math


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 10
N_BINS = 20
NORMALIZE = True

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features_confirmed = []
targets_confirmed = []

death = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_global.csv')
death = data.load_csv_data(death)
features_death = []
targets_death = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features_confirmed.append(cases)
    targets_confirmed.append(labels)

    df2 = data.filter_by_attribute(
        death, "Country/Region", val)
    cases_d, labels_d = data.get_cases_chronologically(df2)
    features_death.append(cases_d)
    targets_death.append(labels_d)


features_confirmed = np.concatenate(features_confirmed, axis=0)
targets_confirmed = np.concatenate(targets_confirmed, axis=0)

features_death = np.concatenate(features_death, axis=0)
targets_death = np.concatenate(targets_death, axis=0)

num_countries = targets_confirmed.shape[0]
death_rate = np.zeros(features_confirmed.shape)

for index_country in range(num_countries):
    cases = features_confirmed[index_country]
    deaths = features_death[index_country]
    death_rate[index_country] = np.array([float(x)/y if y != 0 else 0 for x, y in zip(deaths, cases)])


#-------- kmeans on death rate -----------
too_weird_death_rate = np.any(death_rate > 0.5)
death_rate = (death_rate[too_weird_death_rate])[0]

targets_death = targets_death[too_weird_death_rate]
kmeans = KMeans(n_clusters=10)
kmeans.fit(death_rate)
y_kmeans = kmeans.predict(death_rate)

'''

#------------ knn on death rate ---------------
predictions = {}

for _dist in ['minkowski', 'manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        if val == 'United Kingdom':
            df = df[df['5/19/20'] == 248818]
        cases, labels = data.get_cases_chronologically(df)

        mask = targets_death[:, 1] != val
        tr_features = death_rate[mask]
        tr_targets = death_rate[mask][:, 1]

        too_weird_death_rate = np.any(tr_features > 0.5)
        tr_features = tr_features[too_weird_death_rate]
        tr_targets = tr_targets[too_weird_death_rate]
        
        cases = cases.sum(axis=0, keepdims=True)
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(death_rate, targets_death[:,1])

        label = knn.predict(cases)

        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()

with open('results/death_knn.json', 'w') as f:
    json.dump(predictions, f, indent=4)
'''