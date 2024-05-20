import os
from feature_generation import prepare_features, combine_stocks, ml_preprocessing
from models import model_LinearRegression
from ploting import plot_data
from sklearn.metrics import mean_absolute_error

# Aktien auswählen
relevant_stocks = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'BRK-B', 'LLY', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'MRK', 'ORCL', 'CVX', 'BAC', 'KO', 'CRM', 'NFLX', 'PEP', 'AMD', 'TMO', 'ADBE', 'WFC', 'LIN', 'QCOM', 'CSCO', 'MCD', 'ACN', 'DIS', 'DHR', 'ABT', 'INTU', 'GE', 'CAT', 'AMAT', 'AXP', 'TXN', 'VZ', 'AMGN', 'PFE', 'MS', 'CMCSA', 'IBM', 'NEE', 'UNP']

# Feature-Generation für jede Aktie
for symbol in relevant_stocks:
    file_path = os.path.join('data', 'stock_dataframes', f'{symbol}.csv')
    if not os.path.exists(file_path): # nicht nochmals erstellen wenn schon vorhanden
        prepare_features(symbol, option_volume=False)

# Zusammenführen, Clean-up NaN-Values (im Normalfall 1 Zeile), One-Hot Encoding der kategorischen Variable 'Sector', abspeichern 
stock_df = combine_stocks(relevant_stocks, return_df=True)

# Vorbereitung des Trainings-, Validierungs- und Testdatensatzes
X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, df_train, df_valid, df_test = ml_preprocessing(stock_df)

print("Training...")
model_LinearRegression.fit(X_train_scaled, y_train)

y_pred_train = model_LinearRegression.predict(X_train_scaled)
y_pred_valid = model_LinearRegression.predict(X_valid_scaled)
y_pred_test = model_LinearRegression.predict(X_test_scaled)

###Results###
print("Linear Regression ----------------------------------------------------")
print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))

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
print("Done")
#mae