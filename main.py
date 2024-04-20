from data_generator import download_stock_df, rsi_calculation
from pipeline import pipeline
from models import model_LinearRegression, RNNModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'AMD', 'QCOM', 'IBM']
    df = download_stock_df(tech_stocks)

    
    X_train, y_train, X_valid, y_valid, X_test, y_test = pipeline(df)
    
    model_LinearRegression.fit(X_train, y_train)
    
    
    
    #mae
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, model_LinearRegression.predict(X_train)))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, model_LinearRegression.predict(X_valid)))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, model_LinearRegression.predict(X_test)))
    
    RNNModel = RNNModel(X_train.shape[1])
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    RNNModel.model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_valid, y_valid))
    
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, RNNModel.model.predict(X_train)))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, RNNModel.model.predict(X_valid)))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, RNNModel.model.predict(X_test)))

    
    
    
    