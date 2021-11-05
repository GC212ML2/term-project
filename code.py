import pandas as pd
from preprocess import csv_to_dataframe
from sklearn.preprocessing import LabelEncoder
import FBClassifier
import FBClustering



# df, dfs = csv_to_dataframe("./data/Google-Playstore.csv")

# 테스트 시, 파일 읽기 속도 개선을 위해 미리 결과 출력 후 읽어옴
dfs = pd.read_csv("./tmp/dfs.csv", index_col=0)





# 추가 정리
dfs.drop(["index"], axis=1, inplace=True)

print(dfs)
print(dfs.info())




# print(dfs['Category'].drop_duplicates().tolist())
# 카테고리가 너무 많이서 나중에 전처리를 하거나 아예 드롭해야 할 수도 있을 것 같습니다...ㅠ
# ['Entertainment', 'Food & Drink', 'Action', 'Books & Reference', 'Music & Audio', 'Communication', 'Social', 'Business', 'Strategy', 'Tools', 'Education', 
# 'Casual', 'Health & Fitness', 'Medical', 'Libraries & Demo', 'Productivity', 'Auto & Vehicles', 'Lifestyle', 'Video Players & Editors', 'Personalization', 
# 'Simulation', 'News & Magazines', 'Adventure', 'Finance', 'Arcade', 'Role Playing', 'House & Home', 'Shopping', 'Maps & Navigation', 'Trivia', 'Travel & Local', 
# 'Beauty', 'Photography', 'Weather', 'Sports', 'Art & Design', 'Parenting', 'Racing', 'Puzzle', 'Events', 'Music', 'Dating', 'Educational', 'Comics', 'Board', 'Card', 'Word', 'Casino']
dfs.drop(["Category"], axis=1, inplace=True)





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



# Last Updated
## 날짜데이터
### 대기중
dfs.drop(["Last Updated"], axis=1, inplace=True)


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
    # "Category" : dfs["Category"],
    "Rating Count" : dfs["Rating Count"],
    "Maximum Installs" : dfs["Maximum Installs"],
    "Free" : lbl_free,
    # "Last Updated" : dfs["Last Updated"],
    "Content Rating" : lbl_content_rating,
    "Ad Supported" : lbl_ad,
    "In App Purchases" : lbl_in_app_purchase,
    "Editors Choice" : lbl_editors_choice,
    "Price" : lbl_price,
    "Rating" : dfs["Rating"],
})


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


clustering_result = FBClustering.brute_force(X)
print(clustering_result.best_params_)