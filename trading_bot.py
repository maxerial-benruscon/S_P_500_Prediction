import pandas as pd
import os
from feature_generation import prepare_features, combine_stocks, ml_preprocessing
import plotly.express as px
import yfinance as yf


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


model_name = "Linear Regression"
predictions_path = os.path.join("data", f"{model_name}_y_pred_test.csv")
predictions_df = pd.read_csv(predictions_path, usecols=['0'])
predictions_df.rename(columns={'0': 'Prediction'}, inplace=True)

trading_df = df_test[['Symbol', 'Open', 'Close']].copy()
trading_df['Prediction_for_tomorrow'] = predictions_df['Prediction'].values

single_stock_df_list = []

relevant_stocks = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'BRK-B', 'LLY', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'MRK', 'ORCL', 'CVX', 'BAC', 'KO', 'CRM', 'NFLX', 'PEP', 'AMD', 'TMO', 'ADBE', 'WFC', 'LIN', 'QCOM', 'CSCO', 'MCD', 'ACN', 'DIS', 'DHR', 'ABT', 'INTU', 'GE', 'CAT', 'AMAT', 'AXP', 'TXN', 'VZ', 'AMGN', 'PFE', 'MS', 'CMCSA', 'IBM', 'NEE', 'UNP']

for symbol in relevant_stocks:

    single_stock_df = trading_df[trading_df['Symbol']==symbol].copy()
    single_stock_df['Prediction_for_today'] = single_stock_df['Prediction_for_tomorrow'].shift(1)
    single_stock_df.drop(single_stock_df.head(1).index, inplace=True)

    single_stock_df['Difference_open_to_prediction_today'] = single_stock_df['Prediction_for_today'] - single_stock_df['Open']

    single_stock_df_list.append(single_stock_df)

trading_df = pd.concat(single_stock_df_list)


trading_days = trading_df.index.unique()

# dummy_trading_date = trading_days[1]

account = 1000000
n_stocks = 5
budget_per_stock = int(account / n_stocks)

daily_account_change = []
trades = []


for trading_day in trading_days:

    trading_day_df = trading_df.loc[trading_df.index == trading_day]

    top_stock_df = trading_day_df.nlargest(n_stocks, 'Difference_open_to_prediction_today')

    account_before_trades = account
    # daily_account_change = []

    # trades = []
    for i in range(n_stocks):
        symbol = top_stock_df.iloc[i]['Symbol']
        open = top_stock_df.iloc[i]['Open']
        close = top_stock_df.iloc[i]['Close']

        units_bought = budget_per_stock // open
        money_spent = units_bought * open
        account -= money_spent

        money_earned = units_bought * close
        account += money_earned

        abs_gain = money_earned - money_spent
        rel_gain = (close - open) / open

        trades.append(
            {
                'Date' : trading_day,
                'Symbol' : symbol,
                'Abs_gain' : abs_gain,
                'Rel_gain' : rel_gain
            }
        )

    account_after_trades = account
    abs_account_change = account_after_trades - account_before_trades
    rel_account_change = (account_after_trades - account_before_trades) / account_before_trades

    daily_account_change.append(
        {
            'Date' : trading_day,
            'Account' : account,
            'Abs_change' : abs_account_change,
            'Rel_change' : rel_account_change
        }
    )

trades_df = pd.DataFrame(trades)
daily_account_change_df = pd.DataFrame(daily_account_change)


# Create a line plot using Plotly
fig = px.line(daily_account_change_df, x='Date', y='Account', title='Portfolio Value in USD',
              labels={'Date': 'Date', 'Account': 'Portfolio Value ($)'})

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_tickformat='%Y-%m-%d',
    template='plotly_white'
)

# Save the plot as HTML file to display
fig.write_html("plots/portfolio_value.html")


# Create a line plot using Plotly
fig = px.line(daily_account_change_df, x='Date', y='Abs_change', title='Absolute Change of the Portfolio Over Time',
              labels={'Date': 'Date', 'Abs_change': 'Absolute Change ($)'})

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Absolute Change ($)',
    xaxis_tickformat='%Y-%m-%d',
    template='plotly_white'
)

# Save the plot as HTML file to display
fig.write_html("plots/portfolio_absolute_change.html")


# Define the period for which you want the S&P 500 data
start_date = "2015-01-05"
end_date = "2022-12-30"

# Download the S&P 500 data
sp500_all_data_df = yf.download('^GSPC', start=start_date, end=end_date)

# Keep only the closing price
sp500_df = sp500_all_data_df[['Close']].copy()

# Rename the column to 'SP500'
sp500_df.rename(columns={'Close': 'SP500'}, inplace=True)


daily_account_change_df['Date'] = pd.to_datetime(daily_account_change_df['Date'])
daily_account_change_df.set_index('Date', inplace=True)

# Merge the two dataframes on the Date column
combined_df = daily_account_change_df.merge(sp500_df, left_index=True, right_index=True)


# Normalize both columns to start at 100 for comparison
combined_df['Portfolio'] = combined_df['Account'] / combined_df['Account'].iloc[0] * 100
combined_df['SP500'] = combined_df['SP500'] / combined_df['SP500'].iloc[0] * 100


# Create a line plot using Plotly
fig = px.line(combined_df, x=combined_df.index, y=['Portfolio', 'SP500'], title='Portfolio vs S&P 500',
              labels={'value': 'Normalized Value', 'index': 'Date', 'variable': 'Legend'})

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Normalized Value',
    xaxis_tickformat='%Y-%m-%d',
    template='plotly_white'
)

# Save the plot as HTML file to display
fig.write_html("plots/portfolio_vs_sp500.html")

# Sharpe Ratio

# Calculate mean and standard deviation of daily returns
mean_daily_return = daily_account_change_df['Rel_change'].mean()
std_daily_return = daily_account_change_df['Rel_change'].std()

# Assuming a risk-free rate of 0 for simplicity
risk_free_rate = 0.01

# Calculate the Sharpe Ratio
sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Mean Daily Return: {mean_daily_return}")
print(f"Portfolio STD: {std_daily_return}")

# Calculate daily returns
daily_account_change_df['Daily_Return'] = daily_account_change_df['Account'].pct_change()

# Remove the first row which will be NaN due to pct_change
daily_account_change_df = daily_account_change_df.dropna(subset=['Daily_Return'])

# Calculate the average daily return and standard deviation of daily returns
average_daily_return = daily_account_change_df['Daily_Return'].mean()
std_daily_return = daily_account_change_df['Daily_Return'].std()

# Assume risk-free rate of 2% per annum
risk_free_rate_annual = 0.02
risk_free_rate_daily = risk_free_rate_annual / 252  # Assuming 252 trading days in a year

# Calculate Sharpe ratio
sharpe_ratio = (average_daily_return - risk_free_rate_daily) / std_daily_return
annualised_sharpe_ratio = (252**0.5) * sharpe_ratio
print(f"Daily Sharpe Ratio: {sharpe_ratio}")
print(f"Annualised Sharpe Ratio: {annualised_sharpe_ratio}")


# Which stocks were often traded?
print(trades_df['Symbol'].value_counts())