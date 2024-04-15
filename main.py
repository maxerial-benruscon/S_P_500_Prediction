from data_generator import download_stock_data
from pipeline import preprocess_data
from models import ReferenceModel, DeepModel, CNNModel
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = download_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    x_train, y_train, x_test, y_test, scaler = preprocess_data(df)
    #convert to pandas dataframe
    model = ReferenceModel(input_shape=(x_train.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(x_train, y_train, batch_size=10, epochs=1, validation_data=(x_test, y_test))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    train_rmse = mrmse = np.sqrt(np.mean(((predictions - y_train) ** 2)))
    test_rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    
    print('Train RMSE:', train_rmse)
    print('Test RMSE:', test_rmse)
    
    DeepModel = DeepModel(input_shape=(x_train.shape[1], 1))
    DeepModel.compile(optimizer='adam', loss='mean_squared_error')
    DeepModel.train(x_train, y_train, batch_size=10, epochs=1, validation_data=(x_test, y_test))
    
    predictions = DeepModel.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    train_rmse = np.sqrt(np.mean(((predictions - y_train) ** 2)))
    test_rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    
    CNNModel = CNNModel(input_shape=(x_train.shape[1], 1))
    CNNModel.compile(optimizer='adam', loss='mean_squared_error')
    CNNModel.train(x_train, y_train, batch_size=10, epochs=1, validation_data=(x_test, y_test))
    
    predictions = CNNModel.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    train_rmse = np.sqrt(np.mean(((predictions - y_train) ** 2)))
    test_rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    
    
    
    