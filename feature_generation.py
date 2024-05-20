import pandas as pd
import yfinance as yf
from datetime import datetime
import pandas_ta as ta
import os
from sklearn.preprocessing import MinMaxScaler

def get_indicators(symbol, start, end):

    # Indikatoren festlegen
    momentum_indicators = ["rsi", "macd", "stoch", "mom", "tsi", "adx"]
    volume_indicators = ["obv", "vwap", "pvo", "ad", "mfi", "cmf"]
    volatility_indicators = ["bbands", "atr", "kc"] # Ichimoku separat wegen potenziellem Data Leakage
    candlestick_patterns = ["hammer", "morningstar", "hangingman", "darkcloudcover", "engulfing", "doji"] # Patterns "3whitesoldiers" und "3blackcrows" sind zu selten, entfernt

    # Stock Data laden
    df = yf.download(symbol, start, end)

    # Indikatoren berechnen
    for momentum_indicator in momentum_indicators:
        df.ta(kind=momentum_indicator, append=True)
    
    for volume_indicator in volume_indicators:
        df.ta(kind=volume_indicator, append=True)
    
    for volatility_indicator in volatility_indicators:
        df.ta(kind=volatility_indicator, append=True)

    # Separat wegen Sub-Indikator mit potenziellem Data Leakage: 'ICS_26'
    ichimoku = ta.ichimoku(high=df['High'], low=df['Low'], close=df['Close'], lookahead=False)
    ichimoku_visible = ichimoku[0]
    df = pd.concat([df, ichimoku_visible], axis=1)

    # Candlestick Patterns
    patterns_df = df.ta.cdl_pattern(name=candlestick_patterns) #cdl_pattern generiert separates df
    df = pd.concat([df, patterns_df], axis=1)

    # Binary Encoding der Candlestick Patterns
    bullish_patterns = ['CDL_HAMMER', 'CDL_MORNINGSTAR'] # Werte sind 0 oder 100
    bearish_patterns= ['CDL_HANGINGMAN', 'CDL_DARKCLOUDCOVER'] # Werte sind 0 oder -100
    bull_bear_patterns = ['CDL_ENGULFING'] # Werte sind -100, 0, 100
    continuation_patterns = ['CDL_DOJI_10_0.1'] # Werte sind 0 oder 100
    candlestick_columns = bullish_patterns + bearish_patterns + bull_bear_patterns + continuation_patterns

    for bullish_pattern in bullish_patterns:
        df[f'{bullish_pattern}_bullish'] = (df[bullish_pattern] == 100).astype(int)

    for bearish_pattern in bearish_patterns:
        df[f'{bearish_pattern}_bearish'] = (df[bearish_pattern] == -100).astype(int)

    for bull_bear_pattern in bull_bear_patterns:
        df[f'{bull_bear_pattern}_bullish'] = (df[bull_bear_pattern] == 100).astype(int)
        df[f'{bull_bear_pattern}_bearish'] = (df[bull_bear_pattern] == -100).astype(int)

    for continuation_pattern in continuation_patterns:
        df[f'{continuation_pattern}_continuation'] = (df[continuation_pattern] == 100).astype(int)

    df.drop(columns=candlestick_columns, inplace=True) # ursprüngliche Candlestick Spalten nicht mehr nötig


    return df


def add_interest(df):
    interest_path = os.path.join('data', 'DFF.csv')
    interest_df = pd.read_csv(interest_path, sep=',', encoding='UTF-8', parse_dates=['DATE'], index_col='DATE')
    df = pd.merge(df, interest_df, left_index=True, right_index=True) # Left-Join auf dem Index <-- Datum

    return df


def add_vix(df):
    vix_path = os.path.join('data','VIX_History.csv')
    vix_df = pd.read_csv(vix_path, sep=',', encoding='UTF-8', parse_dates=['DATE'], index_col='DATE')

    vix_column_renaming = {
        'OPEN':'VIX_open',
        'CLOSE':'VIX_close',
        'HIGH':'VIX_high',
        'LOW':'VIX_low'
    }
    vix_df.rename(columns=vix_column_renaming, inplace=True)

    # Berechnung absolute und relative Veränderung
    vix_df['VIX_abs_change'] = vix_df['VIX_close'] - vix_df['VIX_open']
    vix_df['VIX_rel_change'] = vix_df['VIX_abs_change'] / vix_df['VIX_open']

    df = pd.merge(df, vix_df, left_index=True, right_index=True)

    return df


def add_s_p_500(df, start, end):
    gspc_df = yf.download('^GSPC', start, end)
    gspc_df.drop(columns=['Adj Close', 'Volume'], inplace=True)
    gspc_column_renaming = {
        'Open':'S_P_500_open',
        'Close':'S_P_500_close',
        'High':'S_P_500_high',
        'Low':'S_P_500_low'
    }
    gspc_df.rename(columns=gspc_column_renaming, inplace=True)

    # Berechnung absolute und relative Veränderung
    gspc_df['S_P_500_abs_change'] = gspc_df['S_P_500_close'] - gspc_df['S_P_500_open']
    gspc_df['S_P_rel_change'] = gspc_df['S_P_500_abs_change'] / gspc_df['S_P_500_open']

    df = pd.merge(df, gspc_df, left_index=True, right_index=True)

    return df


def add_option_volume(df, stock):

    stock = stock.replace('-', '') # Beispiel BRK-B -> BRKB

    #######################
    #       Monthly       #
    #######################

    # von 2003 bis Ende 2006 ist nur das durchschnittliche tägliche Optionsvolumen jedes Monats verfügbar
    
    # DataFrame erstellen, wo für jeden Monat das tägliche Durchschnittsvolumen in jeden Tag geschrieben wird
    date_range = pd.date_range(start='2003-01-01', end='2006-12-31')
    monthly_option_df = pd.DataFrame(index=date_range, columns=['Option Volume'])

    options_path_monthly =os.path.join('data', 'options', 'monthly')

    for year in range(2003, 2006+1):
        year_str = str(year)

        for month in range(1, 12+1):

            month_str = str(month).zfill(2) # Null auffüllen wo nötig, beispiel '1' -> '01'

            if year < 2013:  # Datenformat von 2003 bis 2012 ist .xls, danach .xlsx
                file_extension = '.xls'
            else:
                file_extension = '.xlsx'
            
            file_name = f'{year}_{month_str}_rank_wosym{file_extension}'
            specific_path = os.path.join(options_path_monthly, file_name)

            options_df = pd.read_excel(specific_path) # Dependencies neben Pandas: openpyxl und xlrd <-- pip install

            # Spalten werden anfangs klein geschrieben...
            if year == 2003 or (year == 2004 and month in [1,2,3]):
                symbol_column = 'symbol'
            else:
                symbol_column = 'Symbol'

            average_daily_volume = options_df.loc[options_df[symbol_column] == stock, 'Tot ADV'].values

            if len(average_daily_volume) > 0:  # Check if data exists for the stock in that month
                # Fill 'Option Volume' column for the month with the extracted value
                start_date = pd.Timestamp(f'{year}-{month_str}-01')
                end_date = pd.Timestamp(start_date.year, start_date.month, start_date.days_in_month)
                monthly_option_df.loc[start_date:end_date, 'Option Volume'] = average_daily_volume[0]


    #######################
    #        Daily        #
    #######################

    # Tägliches Optionsvolumen ab 2007 verfügbar
    yearly_dfs = []
    options_path_yearly = os.path.join('data', 'options', 'daily')

    for year in range(2007, 2022+1):
        year_str = str(year)
        specific_path = os.path.join(options_path_yearly, f'daily_volume_{year_str}.csv')

        options_df = pd.read_csv(specific_path, sep=',', encoding='UTF-8', parse_dates=['Trade Date'])

        stock_df = options_df[options_df['Underlying'] == stock]

        # Nach Datum gruppieren und anschliessend das Volumen der gruppierten Zeilen summieren
        daily_volume_sum = stock_df.groupby('Trade Date')['Average Daily Volume'].sum().reset_index()

        daily_volume_sum.rename(columns={'Average Daily Volume':'Option Volume'}, inplace=True)

        yearly_dfs.append(daily_volume_sum)

    daily_option_df = pd.concat(yearly_dfs, ignore_index=True)
    daily_option_df.set_index('Trade Date', inplace=True)



    #######################
    #       Zusammen      #
    #######################

    option_df = pd.concat([monthly_option_df, daily_option_df])

    df = pd.merge(df, option_df, left_index=True, right_index=True)

    return df


def add_yfinance(df, stock):
    ticker = yf.Ticker(stock)
    df['Symbol'] = stock
    df['Sector'] = ticker.info['sector']
    df['Industry'] = ticker.info['industry']

    return df


def add_change(df):
    df['Absolute change'] = df['Close'] - df['Open']
    df['Relative change'] = df['Absolute change'] / df['Open']

    df.drop(columns=['Adj Close'], inplace=True)

    return df


def rearrange_columns(df):
    columns = df.columns.tolist()

    # Remove 'Absolute change' and 'Relative change' from their original positions
    columns.remove('Absolute change')
    columns.remove('Relative change')

    # Insert 'Absolute change' and 'Relative change' at the 5th and 6th positions respectively
    columns.insert(4, 'Absolute change')
    columns.insert(5, 'Relative change')

    # Reorder the columns in the DataFrame
    df = df[columns]
    return df


def add_next_close(df):
    df['Next close'] = df['Close'].shift(-1) # Zielvariable = Close des nächsten Tages
    df.drop(df.tail(1).index, inplace=True) # Letze Zeile hat bei 'Next Close' NaN -> droppen
    return df


def save_df(df, stock, option_volume):

    if option_volume:
        folder_path = os.path.join('data', 'stock_dataframes_OV')
    
    else:
        folder_path = os.path.join('data', 'stock_dataframes')

    # Falls Ordner noch nicht besteht, erstellen
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    

    file_path = os.path.join(folder_path, f'{stock}.csv')
    df[77:].to_csv(file_path, index=True, index_label='Date') # Ab Zeile 77 wegen Einschwingphase einiger Indikatoren
    

def get_s_p_stocks(threshold):

    s_p_path = os.path.join('data', 'sp500_companies.csv')
    s_p_df = pd.read_csv(s_p_path, sep=',', encoding='UTF-8')
    s_p_reduced = s_p_df[['Symbol', 'Weight']]

    if threshold == 100:
        relevant_stocks = s_p_reduced['Symbol'].tolist()
        return relevant_stocks

    weight_sum = 0
    relevant_stocks = []

    for i in range(s_p_reduced.shape[0]):
        if weight_sum*100 >= threshold:
            return relevant_stocks

        weight_sum += s_p_reduced.iloc[i]['Weight']
        relevant_stocks.append(s_p_reduced.iloc[i]['Symbol'])

    return 'Something went wrong'


def prepare_features(stock_symbol, option_volume=False):
    end = datetime.strptime('2023-01-01', '%Y-%m-%d')
    start = datetime(end.year - 20, end.month, end.day)

    df = get_indicators(stock_symbol, start, end)
    df = add_interest(df)
    df = add_vix(df)
    df = add_s_p_500(df, start, end)
    if option_volume:
        df = add_option_volume(df, stock_symbol)
    df = add_yfinance(df, stock_symbol)
    df = add_change(df)
    df = rearrange_columns(df)
    df = add_next_close(df)
    save_df(df, stock_symbol, option_volume)

    print(f'DataFrame for {stock_symbol} has been prepared and saved')


def combine_stocks(relevant_stocks, save_file_name='combined_dataframe', return_df=False):
    print('Combining Data of all Stocks into one DataFrame...')
    # relevant_stocks = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'BRK-B', 'LLY', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'MRK', 'ORCL', 'CVX', 'BAC', 'KO', 'CRM', 'NFLX', 'PEP', 'AMD', 'TMO', 'ADBE', 'WFC', 'LIN', 'QCOM', 'CSCO', 'MCD', 'ACN', 'DIS', 'DHR', 'ABT', 'INTU', 'GE', 'CAT', 'AMAT', 'AXP', 'TXN', 'VZ', 'AMGN', 'PFE', 'MS', 'CMCSA', 'IBM', 'NEE', 'UNP']

    stock_df_path = os.path.join('data', 'stock_dataframes')

    all_stocks = []

    for symbol in relevant_stocks:
        file_path = os.path.join(stock_df_path, f'{symbol}.csv')
        stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        all_stocks.append(stock_df)

    # Alle DataFrames zusammenführen
    big_df = pd.concat(all_stocks)

    print('Checking for NaN Values...')
    for column in big_df.columns:
        nan_sum = big_df[column].isna().sum()
        if nan_sum > 0:
            print(f'Column {column} has {nan_sum} NaN Value(s)')

    old_shape = big_df.shape
    big_df.dropna(subset=['VWAP_D'], inplace=True)
    print(f'Dropped {old_shape[0] - big_df.shape[0]} Rows. (1 is the bugged AMD Row)')

    # Nach Datum sortieren
    # big_df.sort_values(by=['Date'], inplace=True)
    big_df.sort_values(by=['Date', 'Symbol'], inplace=True)

    # One-Hot Encoding
    # 10 Sektoren zu Spalten. Die 30 Sub-Sektoren ('Industry') wären zu viel
    one_hot_encoded = pd.get_dummies(big_df['Sector'], prefix='Sector')
    one_hot_encoded_numeric = one_hot_encoded.astype(int)
    big_df = pd.concat([big_df, one_hot_encoded_numeric], axis=1)
    big_df.drop(columns=['Sector', 'Industry'], inplace=True)

    # Zielvariable 'Next close' als letzte Spalte
    column_list = big_df.columns.tolist()
    column_list.remove('Next close')
    column_list.append('Next close')
    big_df = big_df[column_list]

    save_path = os.path.join('data', 'ML_data')
    # Falls Ordner noch nicht besteht, erstellen
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, f'{save_file_name}.csv')
    big_df.to_csv(save_file_path, index=True)
    print(f'Saved DataFrame as {save_file_name}.csv at Path {save_file_path}')
    print('-------------------------------------------------------------------------------------')

    if return_df:
        return big_df


def ml_preprocessing(df):

    print('Preparing Train, Validation and Test Data for ML...')

    one_hot_columns = ['CDL_HAMMER_bullish', 'CDL_MORNINGSTAR_bullish','CDL_HANGINGMAN_bearish', 'CDL_DARKCLOUDCOVER_bearish', 'CDL_ENGULFING_bullish', 'CDL_ENGULFING_bearish', 'CDL_DOJI_10_0.1_continuation', 'Sector_Basic Materials', 'Sector_Communication Services', 'Sector_Consumer Cyclical', 'Sector_Consumer Defensive', 'Sector_Energy', 'Sector_Financial Services', 'Sector_Healthcare', 'Sector_Industrials', 'Sector_Technology', 'Sector_Utilities']

    symbol_column = 'Symbol'
    target_variable = 'Next close'

    scale_columns = df.columns.tolist()
    scale_columns = [x for x in scale_columns if x not in one_hot_columns]
    scale_columns.remove(symbol_column)
    scale_columns.remove(target_variable)

    train_columns = scale_columns + one_hot_columns

    df_train = df.loc[df.index < '2010-01-01']
    df_valid = df.loc[(df.index >= '2010-01-01') & (df.index < '2015-01-01')]   
    df_test = df.loc[df.index >= '2015-01-01']

    X_train = df_train[train_columns]
    y_train = df_train[target_variable]

    X_valid = df_valid[train_columns]
    y_valid = df_valid[target_variable]

    X_test = df_test[train_columns]
    y_test = df_test[target_variable]


    # Skalierung der Tabellen mit numerischen Werten (One-Hot Encoding nicht skalieren)
    scaler = MinMaxScaler()

    X_train_scaled = X_train.copy()
    X_train_scaled[scale_columns] = scaler.fit_transform(X_train[scale_columns]) # auf Trainings-Daten Skalieren

    X_valid_scaled = X_valid.copy()
    X_valid_scaled[scale_columns] = scaler.transform(X_valid[scale_columns])

    X_test_scaled = X_test.copy()
    X_test_scaled[scale_columns] = scaler.transform(X_test[scale_columns])

    print('Data Preparation finished')
    print('-------------------------------------------------------------------------------------')

    return X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, df_train, df_valid, df_test

