""" Tax Algo """

import pandas as pd
import numpy as np

def get_last_prices():
    """
    Gets the data from the system"
    returns: A pandas dataframe with the financial data
    """
    data_day = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Last_Prices.csv",index_col=0,header=0)
    
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
    trade_history = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Trade_History.csv",index_col=0,header=0)
    trade_history.index = pd.to_datetime(trade_history.index, format='%m/%d/%Y')
    return(trade_history)
    
def get_dict_index_instrument():
    """
    Gets all the trade history for the client
    returns: The client trade history data
    """
    dict_instrument = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_Inst.csv",index_col=0,header=0)
    
    return(dict_instrument)

def get_inventory_price(trade_history):
    """
    The funtion process a trade history dataframe and then makes a report of a net position for a portfolio
    params: trade_history: A dataframe of the trade history for a given client
    returns: A dataframe with the summary information for the client
    """
    tickers= np.unique(trade_history['Asset'].values)
    columns = ['Quantity','Holding Price','PnL']
    portfolio = pd.DataFrame(index=tickers , columns=columns)
    
    for ticker in tickers:
        ticker_history = trade_history[trade_history['Asset'] == ticker ].sort_index(ascending=True)
        
    return(portfolio)

if __name__ == "__main__":
    data_day = get_last_prices()
    holdings = get_portfolio()
    trade_history = get_trade_history()
    dict_instrument = get_dict_index_instrument()
    portfolio = get_inventory_price(trade_history)
    print(portfolio)