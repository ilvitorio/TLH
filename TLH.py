""" Tax Algo """

import pandas as pd


def get_last_prices():
    "Gets the data from the system"
    "returns: A pandas dataframe with the financial data"
    data_day = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Last_Prices.csv",index_col=0,header=0)
    
    return(data_day)

def get_portfolio():
    "Gets the clients portfolio Holding Data"
    "returns: The clients portfolio data"
    holdings = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Portfolio.csv",index_col=0,header=0)
    
    return(holdings)
    
def get_trade_history():
    "Gets all the trade history for the client"