
from preprocess import csv_to_dataframe
import FBClassifier
import FBClustering

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
from matplotlib import pyplot as plt
import seaborn as sns



# Import file and Preprocessing
df, dfs = csv_to_dataframe("./data/Google-Playstore.csv")
# dfs.drop(["index"], axis=1, inplace=True) 



# Group again with 5 groups
#  - [Entertainment / Productivity / Lifestyle / Game / Education / Welfare]
category_le = LabelEncoder()
lbl_category = category_le.fit_transform(dfs['Category'])



# Bool
# Free, Ad Supported, In App Purchases, Editors Choice
free_le = LabelEncoder()
lbl_free = free_le.fit_transform(dfs['Free'])

ad_le = LabelEncoder()
lbl_ad = ad_le.fit_transform(dfs['Ad Supported'])

in_app_purchase_le = LabelEncoder()
lbl_in_app_purchase = in_app_purchase_le.fit_transform(dfs['In App Purchases'])

editors_choice_le = LabelEncoder()
lbl_editors_choice = editors_choice_le.fit_transform(dfs['Editors Choice'])

# Manual encoding for ordering
# ['Everyone', 'Teen', 'Adults']
# print(dfs['Content Rating'].drop_duplicates().tolist())
content_rating_le = LabelEncoder()
lbl_content_rating = content_rating_le.fit_transform(dfs['Content Rating'])
# print(dfs['Content Rating'])
# print(content_rating_le.classes_)


lbl_price = []
# ['Free', 'Low', 'Mid', 'High']
for i in dfs["Price"]:
# print(dfs['Price'].drop_duplicates().tolist())
    if i == "Free": lbl_price.append(0)
    elif i == "Low": lbl_price.append(1)
    elif i == "Mid": lbl_price.append(2)
    elif i == "High": lbl_price.append(3)

price_list_le = ['Free', 'Low', 'Mid', 'High']


dft = pd.DataFrame({
    "Category" : lbl_category,
    "Rating Count" : dfs["Rating Count"],
    "Maximum Installs" : dfs["Maximum Installs"],
    "Free" : lbl_free,
    "Last Updated" : dfs["Last Updated"],
    "Content Rating" : lbl_content_rating,
    "Ad Supported" : lbl_ad,
    "In App Purchases" : lbl_in_app_purchase,
    "Editors Choice" : lbl_editors_choice,
    "Price" : lbl_price,
    "Rating" : dfs["Rating"],
})

# # Print Label
# print(category_le.classes_)
# print(free_le.classes_)
# print(content_rating_le.classes_)
# print(in_app_purchase_le.classes_)
# print(editors_choice_le.classes_)
# print(price_list_le)

# Plot Heatmap
def heatmap(X, title):
    # Calculate correlation matrix and plot them
    plt.figure(figsize=(12,10))
    plt.title('Heatmap of ' + str(title), fontsize=20)
    g=sns.heatmap(X[X.corr().index].corr(), annot=True, cmap="YlGnBu")

    plt.show()

heatmap(dft, "Heatmap test")



# Split to predictor and predicted featrue
X = dft.drop(["Rating"], axis=1)
y = dft["Rating"]
print(X)
print(y)
print(dft.Rating.value_counts())



# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=len(dft.columns)-1)
fit = bestfeatures.fit(X,dft.Rating)
dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores.columns = ['Col', 'Score']
print(featureScores.nlargest(len(dft.columns)-1, 'Score'))

# # select top 4 best features
dft = dft[['Maximum Installs','Ad Supported','In App Purchases','Rating Count','Rating']]



# Training part
## Classification
classifier_result = FBClassifier.auto_ml(
    X,
    dft["Rating"],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
        LogisticRegression(solver="lbfgs", max_iter=500, multi_class="ovr", class_weight='balanced'),
        LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="ovr", class_weight='balanced'),
        GaussianNB(),
        GradientBoostingClassifier() 
    ],
)

# Print the result
print(classifier_result.best_params)
print('best score :', classifier_result.best_score)
print(FBClassifier.clf_report(X, dft.Rating, classifier_result))
FBClassifier.plot_roc_curve(X, dft.Rating, classifier_result, classifier_result.best_model)



# Plot function for Clustering
def plot_grid(xlist, ylist, colors, title, xlabel, ylabel):
    plt.figure(figsize=(17,17))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    group1x = []
    group1y = []
    group2x = []
    group2y = []
    group3x = []
    group3y = []
    group4x = []
    group4y = []
    
    for i in range(0, len(xlist)):
        color = 'ro'
        if colors[i] == 0: 
            group1x.append(xlist[i])
            group1y.append(ylist[i])
        if colors[i] == 1:
            group2x.append(xlist[i])
            group2y.append(ylist[i])
        if colors[i] == 2:
            group3x.append(xlist[i])
            group3y.append(ylist[i])
        if colors[i] == 3: 
            group4x.append(xlist[i])
            group4y.append(ylist[i])

    plt.plot(group1x, group1y, 'ro', label="Group 0")
    plt.plot(group2x, group2y, 'go', label="Group 1")
    plt.plot(group3x, group3y, 'bo', label="Group 2")
    plt.plot(group4x, group4y, 'yo', label="Group 3")
    
    plt.legend()
    plt.show()







# Training part
## Classification

# start = timer() # Ticker for performance test
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN

clustering_result = FBClustering.auto_ml(
    X,
    cluster_k=[3], # [2,3,4,5,6,7,8,9,10,11,12,13,14]
    models=[
        KMeans(),
        GaussianMixture(),
        MeanShift(bandwidth=2),MeanShift(bandwidth=3),MeanShift(bandwidth=4),MeanShift(bandwidth=5),
        DBSCAN(eps=0.1, min_samples=5),DBSCAN(eps=0.2, min_samples=5),DBSCAN(eps=0.3, min_samples=5),DBSCAN(eps=0.5, min_samples=5),
        DBSCAN(eps=0.1, min_samples=7),DBSCAN(eps=0.2, min_samples=7),DBSCAN(eps=0.3, min_samples=7),DBSCAN(eps=0.5, min_samples=7),
    ],
    scalers=[
        None,
        StandardScaler(), 
        RobustScaler(), 
        MinMaxScaler(), 
        MaxAbsScaler()
    ],
)

# end = timer() # Ticker for performance test
# print("Execution time :", timedelta(seconds=end-start)) # Ticker for performance test

print(clustering_result.best_params)
print(clustering_result.best_score)



# Convert for calculate clustering purity score
yt = []
ytt = y.tolist()
for i in range(0, len(ytt)):
    yt.append(ytt[i]-1)
yp = clustering_result.labels.tolist()

purity_result = FBClustering.purity_score(y_true=yt, y_pred=yp)

plot_grid(dft.iloc[:,0], dft.iloc[:,3], yp, "Clustering Result", "Maximum Installs", "Rating Count")
plot_grid(dft.iloc[:,0], dft.iloc[:,3], dft.iloc[:,4], "Clustering Result", "Maximum Installs", "Rating Count")
