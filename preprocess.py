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

    # Replace 'LastUpdated' to YYYYMMDD format
    if df.columns.tolist().count("Last Updated") != 0:
        df = df.rename(columns={'Last Updated':'LastUpdated'})
        df[['L1', 'L2', 'L3']] = pd.DataFrame(df.LastUpdated.str.split(' ', 3).tolist())

        df['L1'] = df['L1'].replace('Jan','01')
        df['L1'] = df['L1'].replace('Feb','02')
        df['L1'] = df['L1'].replace('Mar','03')
        df['L1'] = df['L1'].replace('Apr','04')
        df['L1'] = df['L1'].replace('May','05')
        df['L1'] = df['L1'].replace('Jun','06')
        df['L1'] = df['L1'].replace('Jul','07')
        df['L1'] = df['L1'].replace('Aug','08')
        df['L1'] = df['L1'].replace('Sep','09')
        df['L1'] = df['L1'].replace('Oct','10')
        df['L1'] = df['L1'].replace('Nov','11')
        df['L1'] = df['L1'].replace('Dec','12')

        df['L2'] = df['L2'].str.slice(start=0, stop=2)

        df['Last Updated'] = df['L3'] + df['L1'] + df['L2']

        df = df.drop(['LastUpdated','L1','L2','L3'], axis=1)





    """"
    Group again with 5 groups
    = Entertainment / Productivity / Lifestyle / Game / Education / Welfare
    """
    if df.columns.tolist().count("Category") != 0:
        # Productivity
        df['Category'] = df['Category'].replace('Productivity','Productivity')
        df['Category'] = df['Category'].replace('Tools','Productivity')
        df['Category'] = df['Category'].replace('Business','Productivity')

        # Lifestyle 
        df['Category'] = df['Category'].replace('Food & Drink','Lifestyle')
        df['Category'] = df['Category'].replace('Auto & vehicles','Lifestyle')
        df['Category'] = df['Category'].replace('House & Home','Lifestyle')
        df['Category'] = df['Category'].replace('Shopping','Lifestyle')
        df['Category'] = df['Category'].replace('Maps & Navigation','Lifestyle')
        df['Category'] = df['Category'].replace('Weather','Lifestyle')
        df['Category'] = df['Category'].replace('Events' ,'Lifestyle')
        df['Category'] = df['Category'].replace('Auto & Vehicles' ,'Lifestyle')
        df['Category'] = df['Category'].replace('Communication','Lifestyle')
        df['Category'] = df['Category'].replace('Finance','Lifestyle')
        df['Category'] = df['Category'].replace('Books & Reference','Lifestyle')
        df['Category'] = df['Category'].replace('Libraries & Demo','Lifestyle')

        # Entertainment
        df['Category'] = df['Category'].replace('Music & Audio','Entertainment')
        df['Category'] = df['Category'].replace('Video Players & Editors','Entertainment')
        df['Category'] = df['Category'].replace('Personalization','Entertainment')
        df['Category'] = df['Category'].replace('News & Magazines','Entertainment')
        df['Category'] = df['Category'].replace('Travel & Local' ,'Entertainment')
        df['Category'] = df['Category'].replace('Beauty' ,'Entertainment')
        df['Category'] = df['Category'].replace('Photography' ,'Entertainment')
        df['Category'] = df['Category'].replace('Art & Design' ,'Entertainment')
        df['Category'] = df['Category'].replace('Dating' ,'Entertainment')
        df['Category'] = df['Category'].replace('Comics' ,'Entertainment')
        df['Category'] = df['Category'].replace('Sports' ,'Entertainment')

        # Game
        # Based on https://www.appbrain.com/stats/android-market-app-categories
        df['Category'] = df['Category'].replace('Action','Game')
        df['Category'] = df['Category'].replace('Adventure','Game')
        df['Category'] = df['Category'].replace('Arcade' ,'Game')
        df['Category'] = df['Category'].replace('Board' ,'Game')
        df['Category'] = df['Category'].replace('Card' ,'Game')
        df['Category'] = df['Category'].replace('Casino' ,'Game')
        df['Category'] = df['Category'].replace('Casual','Game')
        df['Category'] = df['Category'].replace('Music' ,'Game')
        df['Category'] = df['Category'].replace('Puzzle' ,'Game')
        df['Category'] = df['Category'].replace('Racing' ,'Game')
        df['Category'] = df['Category'].replace('Role Playing' ,'Game')
        df['Category'] = df['Category'].replace('Simulation','Game')
        df['Category'] = df['Category'].replace('Strategy','Game')
        df['Category'] = df['Category'].replace('Trivia','Game')
        df['Category'] = df['Category'].replace('Word' ,'Game')
        df['Category'] = df['Category'].replace('Educational','Education')

        # Welfare
        df['Category'] = df['Category'].replace('Health & Fitness','Welfare')
        df['Category'] = df['Category'].replace('Medical','Welfare')
        df['Category'] = df['Category'].replace('Parenting','Welfare')



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
