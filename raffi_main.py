import pandas as pd
import os
from feature_generation import prepare_features, combine_stocks, ml_preprocessing

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