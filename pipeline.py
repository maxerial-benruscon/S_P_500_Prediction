#refactoring of https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

from data_generator import download_stock_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = download_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])


def preprocess_data(df):
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    
    return x_train, y_train, x_test, y_test, scaler







