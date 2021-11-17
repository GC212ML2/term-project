import pandas as pd
from preprocess import csv_to_dataframe
from sklearn.preprocessing import LabelEncoder

import FBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import FBClustering

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from timeit import default_timer as timer
from datetime import timedelta

from matplotlib import pyplot as plt
import seaborn as sns



# df, dfs = csv_to_dataframe("./data/Google-Playstore.csv")
# # dfs.to_csv("dfs.csv")
# 테스트 시, 파일 읽기 속도 개선을 위해 미리 결과 출력 후 읽어옴
dfs = pd.read_csv("./tmp/dfs.csv", index_col=0)
dfs.drop(["index"], axis=1, inplace=True) # 추가 정리

print(dfs)
print(dfs.info())


# print(dfs['Category'].drop_duplicates().tolist())
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

rating_le = LabelEncoder()
lbl_rating = rating_le.fit_transform(dfs['Rating'])

print("=====Content Rating=====")
# Ordering 주려면 직접 안코딩해야 함.
# ['Everyone', 'Teen', 'Adults']
print(dfs['Content Rating'].drop_duplicates().tolist())
content_rating_le = LabelEncoder()
lbl_content_rating = content_rating_le.fit_transform(dfs['Content Rating'])
print(dfs['Content Rating'])
print(content_rating_le.classes_)



lbl_price = []
print("=====Price=====")
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
    "Rating" : lbl_rating,
})

# dft = dft[['Category','Maximum Installs','Ad Supported','In App Purchases','Rating']]

print(dft)
# 라벨 출력
print(category_le.classes_)
print(free_le.classes_)
print(content_rating_le.classes_)
print(in_app_purchase_le.classes_)
print(editors_choice_le.classes_)
print(price_list_le)


# ===============================================================
# ===============================================================
# ===============================================================





X = dft.drop(["Rating"], axis=1)
print(X)

print(dft.Rating.value_counts())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=len(dft.columns)-1)
fit = bestfeatures.fit(X,dft.Rating)
dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores.columns = ['Col', 'Score']
print(featureScores.nlargest(len(dft.columns)-1, 'Score'))

# select top 4 best features
dft = dft[['Maximum Installs','Ad Supported','In App Purchases','Rating Count','Rating']]

classifier_result = FBClassifier.auto_ml(
    X,
    dfs["Rating"],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
        LogisticRegression(solver="lbfgs", max_iter=500, multi_class="ovr", class_weight='balanced'),
        LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="ovr", class_weight='balanced'),
        GaussianNB(),
        GradientBoostingClassifier()
    ],
    max_iter = 30,
)

print(classifier_result.best_params)
print('best score :', classifier_result.best_score)
print(FBClassifier.clf_report(X, dft.Rating, classifier_result))
FBClassifier.plot_roc_curve(X, dft.Rating, classifier_result, classifier_result.best_model)





# Plot Heatmap
# def heatmap(X, title):
#     # Calculate correlation matrix and plot them
#     plt.figure(figsize=(12,10))
#     plt.title('Heatmap of ' + str(title), fontsize=20)
#     g=sns.heatmap(X[X.corr().index].corr(), annot=True, cmap="YlGnBu")

#     plt.show()

# heatmap(dft, "Heatmap test")





# ===============================================================
# ===============================================================
# ===============================================================



# start = timer()

# clustering_result = FBClustering.brute_force(
#     X,
#     # cluster_k=[4],
#     scalers=[
#         None,
#         StandardScaler(), 
#         RobustScaler(), 
#         MinMaxScaler(), 
#         MaxAbsScaler()
#     ],
# )

# end = timer()
# print("Execution time :", timedelta(seconds=end-start))

# print(clustering_result.best_params)
# print(clustering_result.best_score)





# start = timer()

# clustering_result = FBClustering.auto_ml(
#     X, 
#     # cluster_k=[4],
#     scalers=[
#         None,
#         StandardScaler(), 
#         RobustScaler(), 
#         MinMaxScaler(), 
#         MaxAbsScaler()
#     ],
# )

# end = timer()
# print("Execution time :", timedelta(seconds=end-start))

# print(clustering_result.best_params)
# print(clustering_result.best_score)
