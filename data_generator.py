#refactoring of https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()




def download_stock_data(tech_list):
    # Set up End and Start times for data grab
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    company_list = []
    company_name = []

    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)
        company_list.append(globals()[stock])
        company_name.append(stock)

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name

    df = pd.concat(company_list, axis=0)
    print(df.info())
    #print unique company names
    print(df['company_name'].unique())
    return df

#download_stock_data(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])