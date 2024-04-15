from data_generator import download_stock_data
from pipeline import preprocess_data
from models import ReferenceModel
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = download_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    x_train, y_train, x_test, y_test, scaler = preprocess_data(df)
    #convert to pandas dataframe
    model = ReferenceModel(input_shape=(x_train.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(x_train, y_train, batch_size=1, epochs=1, validation_data=(x_test, y_test))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    train_rmse = model.model.evaluate(x_train, y_train, verbose=0)
    test_rmse = model.model.evaluate(x_test, y_test, verbose=0)
    
    print('Train RMSE:', train_rmse)
    print('Test RMSE:', test_rmse)
    
    