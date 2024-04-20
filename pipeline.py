import pandas as pd
from sklearn.preprocessing import StandardScaler

def pipeline(df):
    df.dropna(subset=['Close','3ma', '7ma', '14ma', '21ma', '28ma', 'RSI_14' ], inplace=True)
    
    df_train = df.loc[df.index < '2010-01-01']
    df_valid = df.loc[(df.index >= '2010-01-01') & (df.index < '2015-01-01')]   
    df_test = df.loc[df.index >= '2015-01-01']
    
    

    X_train = df_train[['3ma','7ma', '14ma', '21ma', '28ma', 'RSI_14'   ]]
    y_train = df_train['Close']
    
    X_valid = df_valid[['3ma','7ma', '14ma', '21ma', '28ma', 'RSI_14'   ]]
    y_valid = df_valid['Close']

    X_test = df_test[['3ma','7ma', '14ma', '21ma', '28ma', 'RSI_14'   ]]
    y_test = df_test['Close']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test






