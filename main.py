from data_generator import download_stock_df, rsi_calculation
from pipeline import pipeline
from models import model_Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'AMD', 'QCOM', 'IBM']
    df = download_stock_df(tech_stocks)

    
    X_train, y_train, X_valid, y_valid, X_test, y_test = pipeline(df)
    
    model_Ridge.fit(X_train, y_train)
    
    print("Ridge regression model score on training set: ", model_Ridge.score(X_train, y_train))
    print("Ridge regression model score on validation set: ", model_Ridge.score(X_valid, y_valid))
    print("Ridge regression model score on test set: ", model_Ridge.score(X_test, y_test))
    
    #mae
    print("Mean absolute error on training set: ", mean_absolute_error(y_train, model_Ridge.predict(X_train)))
    print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, model_Ridge.predict(X_valid)))
    print("Mean absolute error on test set: ", mean_absolute_error(y_test, model_Ridge.predict(X_test)))
    
    

    
    
    
    