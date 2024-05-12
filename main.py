from data_generator import download_stock_df
from pipeline import pipeline
from models import model_LinearRegression, model_RandomForest, model_XGB, RNNModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ploting import plot_data
import matplotlib.pyplot as plt

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
    print("Linear Regression ----------------------------------------------------")
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))
    

    
    
    ################################################################################################
    ### BaseLine Model Random Forest
    ################################################################################################
    
    model_RandomForest.fit(X_train, y_train)
    
    y_pred_train = model_RandomForest.predict(X_train)
    y_pred_valid = model_RandomForest.predict(X_valid)
    y_pred_test = model_RandomForest.predict(X_test)
    
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMZN', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='TSLA', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='NVDA', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='INTC', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMD', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='QCOM', model='Random Forest')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='IBM', model='Random Forest')
    
    #mae
    print("Random Forrest ----------------------------------------------------")
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))
    
    
    ################################################################################################
    ### BaseLine Model XGB
    ################################################################################################
    
    
    model_XGB.fit(X_train, y_train)
    
    y_pred_train = model_XGB.predict(X_train)
    y_pred_valid = model_XGB.predict(X_valid)
    y_pred_test = model_XGB.predict(X_test)
    
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMZN', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='TSLA', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='NVDA', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='INTC', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AMD', model='XGB')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='QCOM', model='XGB')
    
    #mae
    print("XGB ----------------------------------------------------")
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
    
    history = RNNModel.model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_valid, y_valid))
    
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
    
    
    
    print("RNN ----------------------------------------------------")
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))
    
    #plotting the loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()