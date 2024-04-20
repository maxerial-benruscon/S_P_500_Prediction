from data_generator import download_stock_df, rsi_calculation
from pipeline import pipeline
from models import model_LinearRegression, RNNModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ploting import plot_data

if __name__ == '__main__':
    
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'AMD', 'QCOM', 'IBM']
    df = download_stock_df(tech_stocks)

    
    X_train, y_train, X_valid, y_valid, X_test, y_test, df_train, df_valid, df_test = pipeline(df)
    
    #######################################################################################
    ### BaseLine Model Linear Regression
    #######################################################################################
    
    model_LinearRegression.fit(X_train, y_train)
    
    y_pred_train = model_LinearRegression.predict(X_train)
    y_pred_valid = model_LinearRegression.predict(X_valid)
    y_pred_test = model_LinearRegression.predict(X_test)
    
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMZN', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='TSLA', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='NVDA', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='INTC', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMD', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='QCOM', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='IBM', model='Linear Regression')
    
    #mae
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))
    
    ################################################################################################
    ### RNN Model
    ################################################################################################
    
    RNNModel = RNNModel(X_train.shape[1])
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    RNNModel.model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_valid, y_valid))
    
    y_pred_train = RNNModel.model.predict(X_train)
    y_pred_valid = RNNModel.model.predict(X_valid)
    y_pred_test = RNNModel.model.predict(X_test)
    
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMZN', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='TSLA', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='NVDA', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='INTC', model='RNN')
    
    
    
    