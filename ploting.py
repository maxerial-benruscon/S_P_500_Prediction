import pandas as pd

import plotly.graph_objects as go

def plot_data(data_train, data_valid, data_test, y_train_pred, y_valid_pred, y_test_pred, symbol='AAPL', model='Linear Regression'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_train.loc[data_train['Symbol'] == symbol].index, y=data_train.loc[data_train['Symbol'] == symbol]['Close'], name='Train_Ground Truth'))
    fig.add_trace(go.Scatter(x=data_valid.loc[data_valid['Symbol'] == symbol].index, y=data_valid.loc[data_valid['Symbol'] == symbol]['Close'], name='Validation_Ground Truth'))
    fig.add_trace(go.Scatter(x=data_test.loc[data_test['Symbol'] == symbol].index, y=data_test.loc[data_test['Symbol'] == symbol]['Close'], name='Test_Ground Truth'))

    fig.add_trace(go.Scatter(x=data_train.loc[data_train['Symbol'] == symbol].index, y=y_train_pred[data_train['Symbol'] == symbol], name='Train prediction'))
    fig.add_trace(go.Scatter(x=data_valid.loc[data_valid['Symbol'] == symbol].index, y=y_valid_pred[data_valid['Symbol'] == symbol], name='Validation prediction'))
    fig.add_trace(go.Scatter(x=data_test.loc[data_test['Symbol'] == symbol].index, y=y_test_pred[data_test['Symbol'] == symbol], name='Test prediction'))

    fig.update_layout(title='AAPL Close price', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.write_html(f'plots/{model}_prediction_{symbol}.html')
