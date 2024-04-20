import pandas as pd
import yfinance as yf
from datetime import datetime

def rsi_calculation(series, periods=14, shift=1):
    delta = series.diff()
    #shift ensures that there occurs no data leakage
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean().shift(shift)
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean().shift(shift)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def download_stock_df(tech_list):
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    company_list = []

    for stock in tech_list:
        df = yf.download(stock, start, end)
        df['Symbol'] = stock  # Ensure 'Symbol' is a column
        company_list.append(df)

    # Concatenate with Symbol as part of the index if that is the intended structure
    df = pd.concat(company_list)
    
    
    for window in [3, 7, 14, 21, 28]:
        df[f'{window}ma'] = df.groupby('Symbol')['Close'].transform(lambda x: x.shift(1).rolling(window).mean())

    # Calculate RSI for each group and ensure the index aligns correctly
    df['RSI_14'] = df.groupby('Symbol')['Close'].transform(lambda x: rsi_calculation(x, 14, 1))
    df['RSI_no_shift'] = df.groupby('Symbol')['Close'].transform(lambda x: rsi_calculation(x, 14, 0)) #only to ensure that the shift is working correctly

    df.sort_values(by=['Date'], inplace=True)
    return df


