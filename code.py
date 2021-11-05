import pandas as pd
from preprocess import csv_to_dataframe
from sklearn.preprocessing import LabelEncoder
import FBClassifier
import FBClustering



# df, dfs = csv_to_dataframe("./data/Google-Playstore.csv")
# dfs.to_csv("dfs.csv")
# 테스트 시, 파일 읽기 속도 개선을 위해 미리 결과 출력 후 읽어옴
dfs = pd.read_csv("./tmp/dfs.csv", index_col=0)
# 추가 정리
dfs.drop(["index"], axis=1, inplace=True)

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
print(len(lbl_price))
lbl_category = []
print("=====Category=====")
# ['Entertainment', 'Productivity', 'Lifestyle', 'Game', 'Education', 'Welfare','Social']
for i in dfs["Category"]:
    if i == "Entertainment": lbl_category.append(0)
    elif i == "Productivity": lbl_category.append(1)
    elif i == "Lifestyle": lbl_category.append(2)
    elif i == "Game": lbl_category.append(3)
    elif i == "Education": lbl_category.append(4)
    elif i == "Welfare": lbl_category.append(5)
    elif i == "Social": lbl_category.append(6)

category_list_le = ['Entertainment', 'Productivity', 'Lifestyle', 'Game', 'Education', 'Welfare','Social']


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
print("dft\n", dft)

print(dft)
# 라벨 출력
print(category_le.classes_)
print(free_le.classes_)
print(content_rating_le.classes_)
print(in_app_purchase_le.classes_)
print(editors_choice_le.classes_)
print(price_list_le)


from sklearn.tree import DecisionTreeClassifier

X = dft.drop(["Rating"], axis=1)
print(X)
classifier_result = FBClassifier.brute_force(X, 
    dft["Rating"],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
    ],
    cv_k=[2,3,4,5,],
)
print(classifier_result.best_params)


clustering_result = FBClustering.brute_force(X, cluster_k=[10])
print(clustering_result.best_params_)

