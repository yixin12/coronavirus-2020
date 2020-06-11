"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?

This one is different in that it uses the difference in cases
from day to day, rather than the raw number of cases.
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import scipy
import matplotlib.pyplot as plt


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []
'''
for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}

for _dist in ['minkowski', 'manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # filter the rest of the data to get rid of the country we are
        # trying to predict
        mask = targets[:, 1] != val
        tr_features = features[mask]
        tr_targets = targets[mask][:, 1]

        above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
        tr_features = np.diff(tr_features[above_min_cases], axis=-1)
        if NORMALIZE:
            tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)

        tr_targets = tr_targets[above_min_cases]

        # train knn
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(tr_features, tr_targets)

        # predict
        cases = np.diff(cases.sum(axis=0, keepdims=True), axis=-1)
        # nearest country to this one based on trajectory
        label = knn.predict(cases)
        
        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()

with open('results/knn_diff.json', 'w') as f:
    json.dump(predictions, f, indent=4)
'''

# ---- Find its 30 geographic neighbors ----
# ---- Then perform knn and predict ----
for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    if df.shape[0] >= 2:
        whole_data = df['5/19/20'].idxmax(axis = 0)
        df = df.loc[[whole_data]]
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}

targets_geo_location = targets[:,2:]
mykd = scipy.spatial.KDTree(targets_geo_location)

for _dist in ['minkowski', 'manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        mycountry_geo_loc = labels[:, 2:]
        dist, ind = mykd.query(mycountry_geo_loc,k=30)
        mycountry_30NN_targets = np.concatenate(targets[ind], axis = 0)
        mycountry_30NN_features = np.concatenate(features[ind], axis = 0)

        # filter the rest of the data to get rid of the country we are
        # trying to predict
        mask = mycountry_30NN_targets[:, 1] != val
        tr_features = mycountry_30NN_features[mask]
        tr_targets = mycountry_30NN_targets[mask][:, 1]

        above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
        tr_features = np.diff(tr_features[above_min_cases], axis=-1)

        if NORMALIZE:
            tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)

        tr_targets = tr_targets[above_min_cases]

        # train knn
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(tr_features, tr_targets)

        # predict
        cases = np.diff(cases.sum(axis=0, keepdims=True), axis=-1)
        # nearest country to this one based on trajectory
        label = knn.predict(cases)

        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()

with open('results/knn_geo__diff_2.json', 'w') as f:
    json.dump(predictions, f, indent=4)



# visualize
for country_key in predictions:
    my_nn = predictions[country_key]['manhattan'][0]
    my_nn_data = features[np.where(targets[:,1] == my_nn)][0]
    non_zeros_cases = my_nn_data[np.nonzero(my_nn_data)]
    non_zeros_cases = np.diff(non_zeros_cases,axis = -1)
    non_zeros_cases_normalized = non_zeros_cases / non_zeros_cases.sum(axis=-1, keepdims=True)


    my_country_data = features[np.where(targets[:,1]==country_key)][0]
    my_country_data = my_country_data[np.nonzero(my_country_data)]
    my_country_data = np.diff(my_country_data, axis=-1)
    my_country_data_normalized = my_country_data / my_country_data.sum(axis=-1, keepdims=True)

    plt.plot(non_zeros_cases_normalized, label = 'nearnest neighbor: ' + my_nn)
    plt.plot(my_country_data_normalized, label = 'country to predict: '+ country_key)
    plt.legend()
    plt.savefig("../graphs/hist/"+country_key+".png")
    plt.close()

