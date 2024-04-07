import yfinance as yf
import pandas as pd
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    yearly_historical_data = stock.history(period="1y")
    yearly_historical_data['daily return'] = yearly_historical_data['Close'].pct_change()
    yearly_historical_data['daily return'].dropna(inplace=True)
    volatility = np.std(yearly_historical_data['daily return'])
    annual_volatility = volatility * np.sqrt(252)
    # todo: semester volatility 
    info['volatility'] = volatility
    info['annual volatility'] = annual_volatility
    info['isin'] = ticker
    return info

def get_stock_list():
    stock_list_path = os.path.join(path, "..", "..", "data", "stock_list.csv")
    stock_list = pd.read_csv(stock_list_path)
    return stock_list

if __name__ == "__main__":
    # print number of stocks in the list
    stock_list = get_stock_list()
    print("Number of stocks in the list: {}".format(len(stock_list)))

    # get stock info for each stock in the list
    stock_info_list = []
    # loop through each stock in the dataframe
    for index, row in stock_list.iterrows():
        ticker = row["isin"]
        stock_info = get_stock_info(ticker)

        try:
            stock_list.loc[index, "sector"] = stock_info["sector"]
        except:
            pass
        
        stock_info_list.append(stock_info)
        print("Stock info for {} is done.".format(ticker))
    # save stock list
    stock_list_path = os.path.join(path, "..", "..", "data", "stock_list.csv")
    stock_list.to_csv(stock_list_path, index=False)

    # convert list of dictionaries to dataframe
    stock_info_df = pd.DataFrame(stock_info_list)
    stock_info_path = os.path.join(path, "..", "..", "data", "stock_info.tsv")
    stock_info_df.to_csv(stock_info_path, index=False, sep='\t')
    print("Stocks info is saved to data/stock_info.tsv")

