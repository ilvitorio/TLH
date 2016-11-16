""" Tax Algo """

import pandas as pd
import numpy as np
import datetime 

def get_last_prices():
    """
    Gets the data from the system"
    returns: A pandas dataframe with the financial data
    """
    data_day = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Last_Prices.csv",index_col=0,header=0)
    data_day.index = pd.to_datetime(data_day.index, format='%m/%d/%Y')
    return(data_day)

def get_portfolio():
    """
    Gets the clients portfolio Holding Data
    returns: The clients portfolio data
    """
    holdings = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Portfolio.csv",index_col=0,header=0)
    
    return(holdings)
    
def get_trade_history():
    """
    Gets all the trade history for the client
    returns: The client trade history data
    """
    dtype_dict= {'Asset':'str', 'Transaction':'str', 'Price':'float','Quantity':'int'}
    trade_history = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Trade_History.csv",index_col=0,header=0,dtype=dtype_dict)
    trade_history.index = pd.to_datetime(trade_history.index, format='%m/%d/%Y')
    return(trade_history)
    
def get_dict_index_instrument():
    """
    Gets all the trade history for the client
    returns: The client trade history data
    """
    dict_instrument = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_Inst.csv",index_col=0,header=0)
    return(dict_instrument)

def get_inventory_price(trade_history,inventory_method='Average'):
    """
    The funtion process a trade history dataframe and then makes a report of a net position for a portfolio
    params: trade_history: A dataframe of the trade history for a given client
    returns: A dataframe with the summary information for the client
    """
    tickers= np.unique(trade_history['Asset'].values)
    columns = ['Quantity','Holding Price', '30_day_Flag']
    portfolio = pd.DataFrame(index=tickers , columns=columns)
    #day_analysis = datetime.datetime.today()
    day_analysis = datetime.datetime(2016,11,16)
    if inventory_method == 'Average':
        for ticker in tickers:
            ticker_history = trade_history[trade_history['Asset'] == ticker ].sort_index(ascending=True)
            buy_operations = ticker_history[ticker_history['Transaction'] == "Buy" ]
            sell_operations = ticker_history[ticker_history['Transaction'] == "Sell" ]
            portfolio_value = buy_operations['Price'].dot(buy_operations['Quantity']) - sell_operations['Price'].dot(sell_operations['Quantity'])
            total_quantity = sum(buy_operations['Quantity']) - sum(sell_operations['Quantity'])
            flag_trade = ticker_history.index[-1] > (day_analysis - datetime.timedelta(days=30) )
            portfolio.ix[ticker,:]=[total_quantity, portfolio_value/total_quantity, flag_trade]        
    return(portfolio)

def get_volatility(data):
    """
    The funtion process a dayly dataset and yields out the volatility of each of the columns
    params: data: A dataframe with financial dayly data
    returns: A dataframe with the volatility of the returns
    """
    returns = (data/data.shift(-1)-1)
    vol = returns.std()
    return(vol)

def extend_portfolio(portfolio,vol_vector):
    

if __name__ == "__main__":
    today = datetime.datetime.today()
    data_day = get_last_prices()
    holdings = get_portfolio()
    trade_history = get_trade_history()
    dict_instrument = get_dict_index_instrument()
    portfolio = get_inventory_price(trade_history)
    vol_vector = get_volatility(data_day)
    
    print(portfolio)