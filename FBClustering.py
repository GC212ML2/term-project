
import copy
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans as CLARANS
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.metrics import silhouette_score



# Change CLARANS result to ScikitLearn result
def clarans_label_converter(labels):
  total_len = 0
  for k in range(0, len(labels)):
    total_len += len(labels[k])

  outList = np.empty((total_len), dtype=int)
  cluster_number = 0
  for k in range(0, len(labels)):
    for l in range(0, len(labels[k])):
      outList[labels[k][l]] = cluster_number
    cluster_number += 1
  return outList



# Purity check for latitude and longitude
def purity_check(X, y_pred):

# Make arbitrary target dataset to calculate score.
# Seperating line to Northern and Souther california -> 35.773
  y = np.array([], dtype=int)
  for i in range(0, len(X)):
    if X.iloc[i, 1] > 35.773:
      y = np.append(y, [1])
    else:
      y = np.append(y, [0])
  return purity_socre(y, y_pred)

# Scoring function through purity check formula
def purity_socre(y_true, y_pred):
  # compute contingency matrix (also called confusion matrix)
  contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
  return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



def brute_force(
    X:DataFrame,
    scalers=[None, StandardScaler()],
    models=[
        KMeans(n_clusters = 2), # n_clusters = k
        # GaussianMixture(), # n_components = k
        DBSCAN(eps=0.5, min_samples=5)
    ],
    cluster_k = [3],
):
    """
    Parameters
    ----------
    - `X`: pandas.DataFrame
      - training dataset.
    - `scalers`: array
      - Scaler functions to scale data. This can be modified by user.
      - `None, StandardScaler()` as default
      - This parameter is compatible with `StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler`.
    - `models`: array
      - Model functions to clustering data. This can be modified by user.
      - KMeans, GaussianMixture, DBSCAN(eps=0.5, min_samples=5) as default with hyperparameters.
      - This parameter is compatible with `KMeans, GaussianMixture, DBSCAN, CLARANS, MeanShift`.
    - `cluster_k`: array
      - Number of cluster. Default value is [3].
      - This can be modified by user.

    Returns
    ----------
    - `best_params_`: dictionary
      - `best_scaler_`: Scaler what has best silhouette score.
      - `best_model_`: Model what has best silhouette score.
      - `best_k_`: Best number of clusters
    - `best_score_`: double
      - Represent the silhouette score of the `best_params`.

    Examples
    ----------
    result = findBestOptions(
      df, 
      models=[
        CLARANS(data=df.to_numpy(), number_clusters=1, numlocal=2, maxneighbor=3),
        GaussianMixture(),
        KMeans(),
        DBSCAN(eps=0.5, min_samples=5),
        MeanShift(bandwidth=bandwidth)
      ],
      scalers=[None,], #StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler() 
      cluster_k = range(2,11)
    )

    # Extract results
    labels = result['labels_']
    best_score = result['best_score_']
    result = result['best_params_']
    best_scaler = result['best_scaler_']
    best_model = result['best_model_']
    best_k = result['best_k_']

    # Print the result of best option
    print("\nBest Scaler: ", end="")
    print(best_scaler)
    print("Best Model: ", end="")
    print(best_model)
    print("Score: ", end="")
    print(best_score)
    print("labels: ", end="")
    print(labels)
    print("k: ", end="")
    print(best_k)
    """

    # Initialize variables
    maxScore = -1.0
    best_scaler = None
    best_model = None
    labels_ = None
    best_k_ = None

    curr_case = 1
    total_case = len(scalers) * len(models) * len(cluster_k)

    # Find best scaler
    for n in range(0, len(scalers)):
        if(scalers[n] != None): 
          X = scalers[n].fit_transform(X)
        
        # Find best model
        for m in range(0, len(models)):

            # Scan once for DBSCAN
            isScaned = False
            
            # Find best k value of CV
            for i in range(0, len(cluster_k)):
                print("Progressing: (",end="")
                print(curr_case,end="/")
                print(total_case,end=")\n")
                curr_case += 1

                # model fitting
                models[m].n_clusters = cluster_k[i]       # for k-Means
                models[m].n_components = cluster_k[i]     # for Gaussian Mixture
                
                
                labels = None
                # calculate silhouette score
                if type(models[m]) == type(CLARANS(X,1,0,0)) :                  
                  models[m] = copy.deepcopy(CLARANS(
                    data=X.to_numpy(), 
                    number_clusters=cluster_k[i],   # CLARANS cluster number setting
                    numlocal=models[m].__dict__['_clarans__numlocal'], 
                    maxneighbor=models[m].__dict__['_clarans__maxneighbor']
                  ))
                  models[m].process()
                  clarans_label = models[m].get_clusters()
                  labels = clarans_label_converter(labels=clarans_label)
                  
                  score_result = silhouette_score(X, labels)
                  
                elif type(models[m]) == type(DBSCAN()) or  type(models[m]) == type(MeanShift()) :
                  if isScaned == True:
                    continue
                  
                  isScaned = True
                  labels = models[m].fit_predict(X)
                  
                  # when cluster nuber is just 1, skip scoring
                  gen_cluster_k = len(pd.DataFrame(labels).drop_duplicates().to_numpy().flatten())
                  if gen_cluster_k <= 1:
                    continue                  
                  score_result = silhouette_score(X, labels)

                else:
                  labels = models[m].fit_predict(X)
                  score_result = silhouette_score(X, labels)



                # if mean value of scores are bigger than max variable,
                # update new options(model, scaler, k) to best options
                if maxScore < score_result:
                    maxScore = score_result
                    best_scaler = copy.deepcopy(scalers[n])
                    best_model = copy.deepcopy(models[m])
                    best_k_ = cluster_k[i]
                    # Calculated by DBSCAN
                    if type(best_model) == type(DBSCAN()) or type(best_model) == type(MeanShift()) : best_k_ = gen_cluster_k 
                    labels_ = copy.deepcopy(labels)



    class res:
        best_params = {}

    res.best_params = {
        'best_scaler': best_scaler,
        'best_model' : best_model,
        'best_k': best_k_,
    }
    res.best_scaler = best_scaler
    res.best_model = best_model
    res.best_k = best_k_

    res.best_score = maxScore
    res.labels = labels_


    # Return value with dictionary type
    return res





def auto_ml():
    logo()











def logo():
    print("")
    print("        /‾‾‾‾‾‾‾\    /‾‾‾‾/  /‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾‾‾/\     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾/\/‾‾‾‾‾/\   ")
    print("       /         \  /    /  /    / /            / /             / /    /                  / /     / /   ")
    print("      /    /\    /\/    /  /    / /\‾‾/    /\‾‾\ /    /‾‾‾/    / /    /    /‾/    /‾/    / /     / /    ")
    print("     /    / /   / /    /  /    / /  ‾/    / /‾‾‾/    /   /    / /    /    / /    / /    / /     / /     ")
    print("    /     ‾‾   / /    /__/    / /   /    / /   /    /___/    / /    /    / /    / /    / /     /_/____  ")
    print("   /    /‾/   / /            / /   /    / /   /             / /    /    / /    / /    / /            /\ ")
    print("  /____/ /___/ /____________/ /   /____/ /   /_____________/ /    /____/ /____/ /____/ /____________/ / ")
    print("  \____\ \___\/\____________\/    \____\/    \_____________\/     \____\ \____\ \____\/ \___________\/  ")
    print("                                                                                    Version: 2021.11.10 ")