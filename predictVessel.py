# -*- coding: utf-8 -*-
"""
Vessel prediction using Gaussian Mixture Model clustering on standardized features. If the
number of vessels is not specified, assume 11 vessels.

@author: Abiral Acharya, Amrit Niraula
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    #Anomaly detection step
    isf = IsolationForest(n_estimators=100, max_samples = 256, contamination=0.009, behaviour= "new" )
    isf.fit(testFeatures)
    y_pred = isf.predict(testFeatures)
    
     #append anomaly factor at the end
    append = np.column_stack((testFeatures, y_pred))
    
    #Pipelining step(Feature Scaling)
    pipeline = make_pipeline(StandardScaler(), KernelPCA())
    features_pipeline=pipeline.fit_transform(append) 
    
    #Gaussian Mixture Model
    gmm = GaussianMixture(n_components=numVessels, covariance_type='full',init_params='kmeans',random_state=100)
    predVessels = gmm.fit_predict(features_pipeline)

    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 11 vessels
    return predictWithK(testFeatures, 11, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    data = loadData('set3noVID.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plt.ion()
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = 11
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    