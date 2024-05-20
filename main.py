import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from feature_generation import prepare_features, combine_stocks, ml_preprocessing
from models import RNNModel, CNNModel
from ploting import plot_data, plot_model_history
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib

def prepare_stock_data(relevant_stocks):
    for symbol in relevant_stocks:
        file_path = os.path.join('data', 'stock_dataframes', f'{symbol}.csv')
        if not os.path.exists(file_path):
            prepare_features(symbol, option_volume=False)
    return combine_stocks(relevant_stocks, return_df=True)

def train_and_evaluate_model(model, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)
    
    print(f"{model_name} ----------------------------------------------------")
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"Mean absolute error on training set: {mae_train}")
    print(f"Mean absolute error on validation set: {mae_valid}")
    print(f"Mean absolute error on test set: {mae_test}")

    return model, {'Model': model_name, 'MAE_train': mae_train, 'MAE_valid': mae_valid, 'MAE_test': mae_test}

def save_shap_plots(model, X_train, model_name):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    print(f"SHAP summary plot for {model_name}")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f'plots/explainer/{model_name.lower()}_summary_plot.png', bbox_inches='tight')
    plt.close()

    print(f"SHAP waterfall plot for a single prediction - {model_name}")
    shap.waterfall_plot(shap_values[0], show=False)
    plt.savefig(f'plots/explainer/{model_name.lower()}_waterfall_plot.png', bbox_inches='tight')
    plt.close()

def reshape_for_rnn(X):
    return X.to_numpy().reshape(X.shape[0], X.shape[1], 1)



def main():
    model_LinearRegression = LinearRegression()
    model_XGB = XGBRegressor(n_estimators=10000, learning_rate=0.01, n_jobs=-1, random_state=187)

    df = pd.DataFrame(columns=['Model', 'MAE_train', 'MAE_valid', 'MAE_test'])

    relevant_stocks = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'BRK-B', 'LLY', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'MRK', 'ORCL', 'CVX', 'BAC', 'KO', 'CRM', 'NFLX', 'PEP', 'AMD', 'TMO', 'ADBE', 'WFC', 'LIN', 'QCOM', 'CSCO', 'MCD', 'ACN', 'DIS', 'DHR', 'ABT', 'INTU', 'GE', 'CAT', 'AMAT', 'AXP', 'TXN', 'VZ', 'AMGN', 'PFE', 'MS', 'CMCSA', 'IBM', 'NEE', 'UNP']

    stock_df = prepare_stock_data(relevant_stocks)
    X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, df_train, df_valid, df_test = ml_preprocessing(stock_df)

    # Linear Regression
    model, results = train_and_evaluate_model(model_LinearRegression, "Linear Regression", X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    save_shap_plots(model, X_train_scaled, "Linear Regression")
    
    plot_data(df_train, df_valid, df_test, model.predict(X_train_scaled), model.predict(X_valid_scaled), model.predict(X_test_scaled), symbol='AAPL', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, model.predict(X_train_scaled), model.predict(X_valid_scaled), model.predict(X_test_scaled), symbol='GOOGL', model='Linear Regression')
    plot_data(df_train, df_valid, df_test, model.predict(X_train_scaled), model.predict(X_valid_scaled), model.predict(X_test_scaled), symbol='MSFT', model='Linear Regression')

    # XGBoost
    model, results = train_and_evaluate_model(model_XGB, "XGBoost", X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test)
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    
    plot_data(df_train, df_valid, df_test, model.predict(X_train_scaled), model.predict(X_valid_scaled), model.predict(X_test_scaled), symbol='AAPL', model='XGBoost')

    # RNN
    X_train_np = reshape_for_rnn(X_train_scaled)
    X_valid_np = reshape_for_rnn(X_valid_scaled)
    X_test_np = reshape_for_rnn(X_test_scaled)

    rnn_model = RNNModel(X_train_scaled.shape[1])
    rnn_history = rnn_model.model.fit(X_train_np, y_train, epochs=100, batch_size=128, validation_data=(X_valid_np, y_valid))

    y_pred_train = rnn_model.model.predict(X_train_scaled)
    y_pred_valid = rnn_model.model.predict(X_valid_scaled)
    y_pred_test = rnn_model.model.predict(X_test_scaled)

    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='RNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='RNN')

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    save_shap_plots(rnn_model.model, X_train_np, "RNN")
    df = pd.concat([df, pd.DataFrame({'Model': ['RNN'], 'MAE_train': [mae_train], 'MAE_valid': [mae_valid], 'MAE_test': [mae_test]})], ignore_index=True)
    pd.DataFrame(rnn_history.history).to_csv('history_rnn.csv')

    rnn_history_df = pd.read_csv('history_rnn.csv')
    plot_model_history(rnn_history_df, "RNN")

    # CNN
    cnn_model = CNNModel(X_train_scaled.shape[1])
    cnn_history = cnn_model.model.fit(X_train_np, y_train, epochs=100, batch_size=128, validation_data=(X_valid_np, y_valid))

    y_pred_train = cnn_model.model.predict(X_train_scaled)
    y_pred_valid = cnn_model.model.predict(X_valid_scaled)
    y_pred_test = cnn_model.model.predict(X_test_scaled)

    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='CNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='CNN')
    plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='CNN')

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    save_shap_plots(cnn_model.model, X_train_np, "CNN")
    
    df = pd.concat([df, pd.DataFrame({'Model': ['CNN'], 'MAE_train': [mae_train], 'MAE_valid': [mae_valid], 'MAE_test': [mae_test]})], ignore_index=True)
    pd.DataFrame(cnn_history.history).to_csv('history_cnn.csv')

    cnn_history_df = pd.read_csv('history_cnn.csv')
    plot_model_history(cnn_history_df, "CNN")


    plt.figure(figsize=(10, 5))
    plt.bar(df['Model'], df['MAE_train'], label='MAE_train')
    plt.bar(df['Model'], df['MAE_valid'], label='MAE_valid')
    plt.bar(df['Model'], df['MAE_test'], label='MAE_test')
    plt.title('Model Comparison')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('plots/results/model_comparison.png')
    plt.close()
    
    print(df)
    
    #Save Linear Regression model

    joblib.dump(model_LinearRegression, 'models/linear_regression.pkl')
    
    #Save XGBoost model
    
    joblib.dump(model_XGB, 'models/xgboost.pkl')
    
    #Save RNN model
    
    rnn_model.model.save('models/rnn_model.h5')
    
    #Save CNN model
    
    cnn_model.model.save('models/cnn_model.h5')

if __name__ == "__main__":
    main()
