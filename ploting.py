import os
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_data(data_train, data_valid, data_test, y_train_pred, y_valid_pred, y_test_pred, symbol='AAPL', model='Linear Regression'):
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_train.loc[data_train['Symbol'] == symbol].index, y=data_train.loc[data_train['Symbol'] == symbol]['Close'], name='Train_Ground Truth'))
    fig.add_trace(go.Scatter(x=data_valid.loc[data_valid['Symbol'] == symbol].index, y=data_valid.loc[data_valid['Symbol'] == symbol]['Close'], name='Validation_Ground Truth'))
    fig.add_trace(go.Scatter(x=data_test.loc[data_test['Symbol'] == symbol].index, y=data_test.loc[data_test['Symbol'] == symbol]['Close'], name='Test_Ground Truth'))

    fig.add_trace(go.Scatter(x=data_train.loc[data_train['Symbol'] == symbol].index, y=y_train_pred[data_train['Symbol'] == symbol], name='Train prediction'))
    fig.add_trace(go.Scatter(x=data_valid.loc[data_valid['Symbol'] == symbol].index, y=y_valid_pred[data_valid['Symbol'] == symbol], name='Validation prediction'))
    fig.add_trace(go.Scatter(x=data_test.loc[data_test['Symbol'] == symbol].index, y=y_test_pred[data_test['Symbol'] == symbol], name='Test prediction'))

    fig.update_layout(title=f'{symbol} Close price', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.write_html(f'plots/{model}_prediction_{symbol}.html')
    

def plot_model_history(history_df, model_name):
    plt.plot(history_df['loss'], label='train')
    plt.plot(history_df['val_loss'], label='validation')
    plt.title(f'{model_name} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'plots/histories/history_loss_{model_name.lower()}.png')
    plt.close()

    plt.plot(history_df['mae'], label='train')
    plt.plot(history_df['val_mae'], label='validation')
    plt.title(f'{model_name} model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'plots/histories/history_mae_{model_name.lower()}.png')
    plt.close()