""" Tax Algo """

import pandas as pd
import numpy as np
import datetime 

def get_ticker_desc():
    """
    Gets the description for the tickers
    returns: A pandas dataframe with ticker type description
    """
    ticker_desc = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Ticker_Desc.csv",index_col=0,header=0)
    return(ticker_desc)
    
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
    Gets all the Instrument Information in the System
    returns: The ticker information in the system
    """
    dict_instrument = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_Inst2.csv",index_col=0,header=0)
    return(dict_instrument)

def get_inventory_price(trade_history,inventory_method='Average'):
    """
    The funtion process a trade history dataframe and then makes a report of a net position for a portfolio
    params: trade_history: A dataframe of the trade history for a given client
    returns: A dataframe with the summary information for the client
    """
    tickers= np.unique(trade_history['Asset'].values)
    columns = ['Quantity','Holding Price', '30_day_flag']
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

def get_financial_metrics(data):
    """
    The funtion process a dayly dataset and yields out the volatility and correlation of each of the columns
    params: data: A dataframe with financial dayly data
    returns: A dataframe with the volatility of the returns
    returns: A dataframe with the correlation of the returns
    """
    returns = (data/data.shift(-1)-1)
    vol = returns.std()
    corr_matrix = returns.corr()
    return(vol,corr_matrix)

def get_costs():
    """
    Gets all the cost related to the investable instruments
    returns: A dataframe with the cost summary table
    """    
    cost_summary = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_Cost.csv",index_col=0,header=0)
    return(cost_summary)    

def best_rank(ticker_list):
    """
    Gets the replacement ranking of the instruments and use it to rank the tickers entered
    returns: The ticker best ranked among the list
    """    
    dict_inst = get_dict_index_instrument()
    best_ranked = dict_inst.ix[ticker_list]['Ranking'].idxmax()
    return(best_ranked)

def get_replacement(portfolio, dict_instrument):
    """
    Gets a ranking of Investable instruments for each index and decide how to replace the current investable
    returns: A dataframe with the portfolio replacements for each holding
    """
  
    #This condition activates the wash sale rule advoiding
    replace_portf = portfolio[portfolio['30_day_flag'] == False].index.values
    #Creates the replacement Data Structure
    order_replace = pd.DataFrame(index=replace_portf , columns=['Replacement'])      
    
    for ticker in replace_portf:
        (asset_class,index_related) = dict_instrument.ix[ticker].values[[0,2]]
        #This condition works out: The wash sale rule selecting a substantially different instrument
        #also it only selects investable instruments
        prospective_tickers = dict_instrument[
        (dict_instrument[ 'Asset_class' ] == asset_class) & \
        (dict_instrument[ 'Index_R' ] != index_related) & \
        (~ dict_instrument[ 'Index_R' ].isnull() )].index.values
        
        #Use an algorithm to select the better replacement (Thinking in the fund selection algorithm)
        order_replace.ix[ticker] = best_rank(prospective_tickers)
    return(order_replace) 

def extend_portfolio(portfolio,vol_vector):
    """
    The funtion process a dayly dataset and yields out the volatility of each of the columns
    params: data: A dataframe with financial dayly data
    returns: A dataframe with the volatility of the returns
    """
    
    pass

if __name__ == "__main__":
    today = datetime.datetime.today()
    ticker_desc = get_ticker_desc()
    data_day = get_last_prices()
    holdings = get_portfolio()
    trade_history = get_trade_history()
    dict_instrument = get_dict_index_instrument()
    portfolio = get_inventory_price(trade_history)
    (vol_vector,corr_matrix) = get_financial_metrics(data_day)
    replacement_pairs = get_replacement(portfolio, dict_instrument)
    print(portfolio)
    
