import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import StratifiedShuffleSplit



def csv_to_dataframe(filename, columns = ['App Id','Developer Website','Developer Email','Privacy Policy','Currency','Developer Id','Scraped Time','Minimum Android']):
    # https://www.kaggle.com/gauthamp10/google-playstore-apps
    df = pd.read_csv(filename)

    if columns.count("Rating") != 0:
        columns.remove("Rating")
        print("Rating is target column, this cannot be deleted.")
        return

    """
    [Preprocessing]
    """
    ### 1. Drop the unnecessary columns
    df = df.drop(columns = columns)

    ### 2. Drop the dirty values
    # Fill NaN of Rating & RatingCount with mean

    
    
    df['Rating']  = df['Rating'].astype(float)
    avg = round(df['Rating'].mean(),1)
    df['Rating'].fillna(avg,inplace=True)

    if df.columns.tolist().count("Rating Count") != 0:
        df['Rating Count']  = df['Rating Count'].astype(float)
        avg = round(df['Rating Count'].mean(),1)
        df['Rating Count'].fillna(avg,inplace=True)

    # Replace the values of 'ContentRating'
    if df.columns.tolist().count("Content Rating") != 0:
        df['Content Rating'] = df['Content Rating'].replace('Unrated',"Everyone")
        df['Content Rating'] = df['Content Rating'].replace('Mature 17+',"Adults")
        df['Content Rating'] = df['Content Rating'].replace('Adults only 18+',"Adults")
        df['Content Rating'] = df['Content Rating'].replace('Everyone 10+',"Everyone")

    # Replace 'Installs' to convert numeric value
    if df.columns.tolist().count("Installs") != 0:
        df.Installs = df.Installs.str.replace(',','')
        df.Installs = df.Installs.str.replace('+','')
        df.Installs = df.Installs.str.replace('Free','0')
        df['Installs'] = pd.to_numeric(df['Installs'])

    """
    Binning (Price) : There are 4 values (Free, Low, Mid, High)
    x = Price

    Free : 0 < x < 0.19
    Low : 0.19 < x < 9.99
    Mid : 9.99 < x < 29.99
    High : 29.99 <= x < 410
    """
    if df.columns.tolist().count("Price") != 0:
        bins = [-0.1,0,9.99,29.99,410] 
        label = ['Free','Low','Mid','High']
        binning = pd.cut(df['Price'], bins, labels=label)
        df = df.drop('Price', axis = 1)
        df['Price'] = binning


    # Drop the rest of NaN values
    df = df.dropna(axis=1)


    """
    Binning (Rating) : There are values of 1-10
    x = Rating

    1 : x < 0.5
    2 : 0.5 <= x < 1.0
    3 : 1.0 <= x < 1.5
    4 : 1.5 <= x < 2.0
    5 : 2.0 <= x < 2.5
    6 : 2.5 <= x < 3.0
    7 : 3.0 <= x < 3.5
    8 : 3.5 <= x < 4.0
    9 : 4.0 <= x < 4.5
    10 : 4.5 <= x
    """
    bins = [-0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1] 
    label = ['1','2','3','4','5','6','7','8','9','10']
    binning = pd.cut(df['Rating'], bins, labels=label)
    df = df.drop('Rating', axis = 1)
    df['Rating'] = binning


    # Target = Rating
    target = ['Rating']
    X = df.drop(target, axis = 1)
    y = df[target]


    # 계층적 샘플링
    split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.01, random_state = 0)

    for train_idx, test_idx in split.split(X, y):
        dataframe = df.loc[train_idx]
        dataframe_sampling = df.loc[test_idx]
    
    dataframe = dataframe.reset_index()
    dataframe_sampling = dataframe_sampling.reset_index()

    return dataframe, dataframe_sampling
