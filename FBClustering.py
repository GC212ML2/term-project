
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
import random


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




# Scoring function through purity check formula
def purity_score(y_true, y_pred):
  # compute contingency matrix (also called confusion matrix)
  contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
  return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)




def brute_force(
    X:DataFrame,
    scalers=[None, StandardScaler()],
    models=[
        KMeans(n_clusters = 2),
        DBSCAN(eps=0.5, min_samples=5)
    ],
    cluster_k = [2,3,4,5,6,7,8,9,10],
):
    """
    Brute Force Search
    ----------
    - Find the best parameter what has the best score.
    - This function use Silhouette score for scoring cluster models.

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
      - This parameter is compatible with `KMeans, DBSCAN`.
    - `cluster_k`: array
      - The umber of cluster. Default value is [2,3,4,5,6,7,8,9,10].
      - This can be modified by user.

    Returns
    ----------
    - `best_params`: dictionary 
      - Dictionary data what has the information of below.
    - `best_scaler`: Scaler what has best silhouette score.
    - `best_model`: Model what has best silhouette score.
    - `best_k`: Best number of clusters
    - `best_score`: double
      - Represent the silhouette score of the `best_params`.
    - `labels`: List
      - The label information of cluster.

    Examples
    ----------
    result = FBClustering.brute_force(
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
    labels = result['labels']
    best_score = result['best_score']
    result = result['best_params']
    best_scaler = result['best_scaler']
    best_model = result['best_model']
    best_k = result['best_k']

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





def auto_ml(
    X:DataFrame,
    scalers=[None, StandardScaler()],
    models=[
        KMeans(n_clusters = 2),
        DBSCAN(eps=0.5, min_samples=5),
    ],
    cluster_k = [2,3,4,5,6,7,8,9,10],
    thresh_score = None,
    max_iter = 50,
):
    """
    Auto ML for Clustering
    ----------
    - Find the best parameter what has the best score.
    - This function use `Auto ML` method. This is similar to the Gradient Descent.
    - This function use memoization technique for faster calculation.
    - This function use Silhouette score for scoring cluster models.

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
      - This parameter is compatible with `KMeans, DBSCAN`.
    - `cluster_k`: array
      - The number of cluster. Default value is [2,3,4,5,6,7,8,9,10].
      - This can be modified by user.
    - `thresh_score`: float
      - Default is None. If, algorithm find the score what is higher than thresh_score, then stop and terminate searching.
    - `max_iter`: integer
      - Default is 50. This is meaning that how many iterations in searching loop.

    Returns
    ----------
    - `best_params`: dictionary 
      - Dictionary data what has the information of below.
    - `best_scaler`: Scaler what has best silhouette score.
    - `best_model`: Model what has best silhouette score.
    - `best_k`: Best number of clusters
    - `best_score`: double
      - Represent the silhouette score of the `best_params`.
    - `labels`: List
      - The label information of cluster.

    Examples
    ----------
    result = FBClustering.auto_ml(
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
    labels = result['labels']
    best_score = result['best_score']
    result = result['best_params']
    best_scaler = result['best_scaler']
    best_model = result['best_model']
    best_k = result['best_k']

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

    logo()
    
    # 0. Calculate length of each paramenter
    scalers_len = len(scalers)
    models_len = len(models)
    cluster_k_len = len(cluster_k)


    # 0. Create memorize table for memoization
    mem_table = [[[0 for col in range( cluster_k_len ) ] for row in range( models_len )] for col in range( scalers_len ) ]
    
    # 0. Create point(theta) vector class
    class Point():
        scalers_idx = 0
        models_idx = 0
        cluster_k_idx = 0
    
    # 0. Initialize gradient value
    gradient_theta1 = 0
    gradient_theta2 = 0
    gradient_theta3 = 0
    
    # 0. Pick arbitrary point (theta1 = p1)
    p1 = Point()
    p1.scalers_idx = random.randrange(0, scalers_len)
    p1.models_idx = random.randrange(0, models_len)
    p1.cluster_k_idx = random.randrange(0, cluster_k_len)

    # 0. Initialize max score point
    max_scalers_idx = 0
    max_models_idx = 0
    max_cluster_k_idx = 0
    max_score = 0
    best_labels = None


    for trial in range(0, max_iter):

        # 1. Check previous gradient value of each theta
        #    and pick another arbitrary point (theta = p2)

        def check_gradient(target_gradient_theta, point_val, max_len):

            result = 0
            if target_gradient_theta > 0 and point_val+1 != max_len:
                # if point_val+1 == max_len => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(point_val + 1, max_len)

            elif target_gradient_theta < 0 and point_val != 0:
                # if point_val == 0 => out of range
                # then, get arbitrary point from 0 to len(target)
                result = random.randrange(0, point_val)
            
            else:
                result = random.randrange(0, max_len)

            return result


        p2 = Point()
        p2.scalers_idx = check_gradient(gradient_theta1, p1.scalers_idx, scalers_len)
        p2.models_idx = check_gradient(gradient_theta2, p1.models_idx, models_len)
        p2.cluster_k_idx = check_gradient(gradient_theta3, p1.cluster_k_idx, cluster_k_len)



        # 2. Calculate score(J(theta)) of each theta(point)
        p1_score = 0
        p2_score = 0
        labels = None


        # Check mem_table if score already has been calculated
        if mem_table[p1.scalers_idx][p1.models_idx][p1.cluster_k_idx] != 0:
            p1_score = mem_table[p1.scalers_idx][p1.models_idx][p1.cluster_k_idx]
        else:
            # model fitting
            models[p1.models_idx].n_clusters = cluster_k[p1.cluster_k_idx]       # for k-Means
            models[p1.models_idx].n_components = cluster_k[p1.cluster_k_idx]     # for Gaussian Mixture

            # calculate silhouette score
            if type(models[p1.models_idx]) == type(CLARANS(X,1,0,0)) :                  
              models[p1.models_idx] = copy.deepcopy(CLARANS(
                data=X.to_numpy(), 
                number_clusters=cluster_k[p1.cluster_k_idx],   # CLARANS cluster number setting
                numlocal=models[p1.models_idx].__dict__['_clarans__numlocal'], 
                maxneighbor=models[p1.models_idx].__dict__['_clarans__maxneighbor']
              ))
              models[p1.models_idx].process()
              clarans_label = models[p1.models_idx].get_clusters()
              labels = clarans_label_converter(labels=clarans_label)
              
              p1_score = silhouette_score(X, labels)
              
            elif type(models[p1.models_idx]) == type(DBSCAN()) or  type(models[p1.models_idx]) == type(MeanShift()) :
              
              labels = models[p1.models_idx].fit_predict(X)
              
              # when cluster nuber is just 1, skip scoring
              gen_cluster_k = len(pd.DataFrame(labels).drop_duplicates().to_numpy().flatten())
              if gen_cluster_k <= 1:
                p1_score = -1
              else:                  
                p1_score = silhouette_score(X, labels)

            else:
              labels = models[p1.models_idx].fit_predict(X)
              p1_score = silhouette_score(X, labels)


            # 2-1. Memoization
            mem_table[p1.scalers_idx][p1.models_idx][p1.cluster_k_idx] = p1_score


        if mem_table[p2.scalers_idx][p2.models_idx][p2.cluster_k_idx] != 0:
            p2_score = mem_table[p2.scalers_idx][p2.models_idx][p2.cluster_k_idx]
        else:
            # model fitting
            models[p2.models_idx].n_clusters = cluster_k[p2.cluster_k_idx]       # for k-Means
            models[p2.models_idx].n_components = cluster_k[p2.cluster_k_idx]     # for Gaussian Mixture

            # calculate silhouette score
            if type(models[p2.models_idx]) == type(CLARANS(X,1,0,0)) :                  
              models[p2.models_idx] = copy.deepcopy(CLARANS(
                data=X.to_numpy(), 
                number_clusters=cluster_k[p2.cluster_k_idx],   # CLARANS cluster number setting
                numlocal=models[p2.models_idx].__dict__['_clarans__numlocal'], 
                maxneighbor=models[p2.models_idx].__dict__['_clarans__maxneighbor']
              ))
              models[p2.models_idx].process()
              clarans_label = models[p2.models_idx].get_clusters()
              labels = clarans_label_converter(labels=clarans_label)
              
              p2_score = silhouette_score(X, labels)
              
            elif type(models[p2.models_idx]) == type(DBSCAN()) or  type(models[p2.models_idx]) == type(MeanShift()) :
         
              labels = models[p2.models_idx].fit_predict(X)
              
              # when cluster nuber is just 1, skip scoring
              gen_cluster_k = len(pd.DataFrame(labels).drop_duplicates().to_numpy().flatten())
              if gen_cluster_k <= 1:
                p1_score = -1
              else:
                p2_score = silhouette_score(X, labels)

            else:
              labels = models[p2.models_idx].fit_predict(X)
              p2_score = silhouette_score(X, labels)


            # 2-1. Memoization
            mem_table[p2.scalers_idx][p2.models_idx][p2.cluster_k_idx] = p2_score



        # Save point parameter what have best score
        if p1_score > p2_score:
            if max_score < p1_score:
                max_scalers_idx = p1.scalers_idx
                max_models_idx = p1.models_idx
                max_cluster_k_idx = p1.cluster_k_idx
                max_score = p1_score
                best_labels = copy.deepcopy(labels)

        if p1_score < p2_score:
            if max_score < p2_score:
                max_scalers_idx = p2.scalers_idx
                max_models_idx = p2.models_idx
                max_cluster_k_idx = p2.cluster_k_idx
                max_score = p2_score
                best_labels = copy.deepcopy(labels)

        # If, score get higher score than thresh, terminate gradient searching
        if thresh_score != None and max_score > thresh_score: break



        # 3. Calcuate gradient of each theta(point).
        #    with using above theta value, set another theta(point).
        change_of_cost = p2_score - p1_score
        change_of_theta1 = p2.scalers_idx - p1.scalers_idx
        change_of_theta2 = p2.models_idx - p1.models_idx
        change_of_theta3 = p2.cluster_k_idx - p1.cluster_k_idx

        # If, attribute of theta1 and theta2 are same, set gradient value to 0 (slope = 0)
        def update_gradient_value(change_of_cost, change_of_theta):
            result_gradient = 0
            if change_of_theta != 0:
                result_gradient = change_of_cost / change_of_theta
            return result_gradient

        gradient_theta1 = update_gradient_value(change_of_cost, change_of_theta1)
        gradient_theta2 = update_gradient_value(change_of_cost, change_of_theta2)
        gradient_theta3 = update_gradient_value(change_of_cost, change_of_theta3)


        # 4. Prepare for next gradient (change theta 1 to new position)
        def set_new_point(gradient_theta, compare1, compare2):
            result_idx = 0
            if gradient_theta > 0:
                result_idx = max([compare1, compare2])
            elif gradient_theta < 0:
                result_idx = min([compare1, compare2])
            else: 
                result_idx = compare1
            return result_idx


        # Set new theta1 for the next calculation
        p1.scalers_idx = set_new_point(gradient_theta1, p1.scalers_idx, p2.scalers_idx)
        p1.models_idx = set_new_point(gradient_theta2, p1.models_idx, p2.models_idx)
        p1.cluster_k_idx = set_new_point(gradient_theta3, p1.cluster_k_idx, p2.cluster_k_idx)



    # Return the result
    class res:
        best_params = {}

    res.best_params = {
        'best_scaler': scalers[max_scalers_idx],
        'best_model' : models[max_models_idx],
        'best_k': cluster_k[max_cluster_k_idx],
    }
    res.best_scaler = scalers[max_scalers_idx]
    res.best_model = models[max_models_idx]
    res.best_k = cluster_k[max_cluster_k_idx]

    res.best_score = max_score
    res.labels = best_labels

    # Return value with res class
    return res





def logo():
    print("")
    print("        /‾‾‾‾‾‾\      /‾‾‾\   /‾‾‾‾/\/‾‾‾‾‾‾‾‾‾‾‾\  /‾‾‾‾‾‾‾‾‾‾‾‾\       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\  /‾‾‾‾\     ")
    print("       /        \    /    /\ /    / /            /\/             /\     /                  /\/     /\    ")
    print("      /    /\    \  /    /  /    / /\‾‾/    /\‾‾\ /    /‾‾‾/    / /    /    /‾/    /‾/    / /     / /    ")
    print("     /    / /    / /    /  /    / /  ‾/    / /‾‾‾/    /   /    / /    /    / /    / /    / /     / /     ")
    print("    /     ‾‾    / /    /__/    / /   /    / /   /    /___/    / /    /    / /    / /    / /     / /      ")
    print("   /    /‾/    / /            / /   /    / /   /             / /    /    / /    / /    / /      ‾‾‾‾‾‾/\ ")
    print("  /____/ /____/ /____________/ /   /____/ /   /_____________/ /    /____/ /____/ /____/ /____________/ / ")
    print("  \____\ \____\/\____________\/    \____\/    \_____________\/     \____\ \____\ \____\/ \___________\/  ")
    print("                                                                    for Clustering / Version: 2021.11.10 ")
