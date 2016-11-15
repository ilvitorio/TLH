import numpy as np
import pandas as pd


#class PredictionCalculator(object):
#    def calculate(observations, predictions, values) -> (ERs, cov):
        


#Home
#observations=pd.read_csv("C:\\Users\\Vitty2\\Documents\\Upwork\\Betasmartz - Black Litterman\\Deliver Datasets\\IndexData.csv",index_col=0,header=0)
#cycle=pd.read_csv("C:\\Users\\Vitty2\\Documents\\Upwork\\Betasmartz - Black Litterman\\Deliver Datasets\\CycleVar.csv",index_col=0,header=0)
#hist_probs=pd.read_csv("C:\\Users\\Vitty2\\Documents\\Upwork\\Betasmartz - Black Litterman\\Deliver Datasets\\predict_probs12.csv",index_col=0,header=0)


#Work
observations = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\IndexData.csv",index_col=0,header=0)
cycle=pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\CycleVar.csv",index_col=0,header=0)
hist_probs=pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\predict_probs12.csv",index_col=0,header=0)
index_type=pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_Index.csv",index_col=0,header=0)
cpi_timeseries = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\CPI.csv",index_col=0,header=0)
order = pd.read_csv("C:\\Users\\U447354\\Documents\\Python Scripts\\Beta\\Dict_order.csv",index_col=0,header=0)


def preprocess_data(level_df):
    "leveldf: The dataframe that is going to be indexed and cleaned"
    level_df.index = pd.to_datetime(level_df.index, format='%d/%m/%Y')
    level_df = level_df.dropna()
    return(level_df)

def returns_dataframe(clean_df,index_type_df,periods=1):
    "leveldf: --"
    returns = np.log(clean_df)
    returns -= returns.shift(periods)
    yield_tickers = index_type_df[index_type_df['Type'] == 'Yield'].index.values
    returns.ix[:,yield_tickers] = clean_df.ix[:,yield_tickers] - clean_df.ix[:,yield_tickers].shift(periods) 
    return(round(returns,13))
    
def dayly_cpi_return(cpi_df,periods=1,scale=12):
    returns_cpi = round( (cpi_df.diff(periods) / cpi_df.shift(periods) + 1) ** (1/scale) - 1 , 13)
    new_index = pd.DatetimeIndex( start=cpi_df.index.values[0], end=cpi_df.index.values[-1], freq ='D')
    dayly_returns = returns_cpi.reindex(new_index)
    dayly_returns.fillna(method='pad',inplace=True)
    dayly_returns.dropna(axis=0,inplace=True)
    return(dayly_returns)

def excess_returns(cpi_df,returns_df):
    excess_returns = returns_df.copy(deep=True)
    for i in excess_returns.index.values:
        excess_returns.ix[i,:] = returns_df.ix[i,:] - np.array( cpi_df.ix[i,:].values)
    return(excess_returns)
    
def merge_cycle_obs(day_df,month_df,string,default = 0):
    day_df[string] = default
    var_merge = month_df[string].unique()
    for i in var_merge:
       index_day = month_df[month_df[string] == i].index.to_datetime()
       for j in index_day:
           date_indexer = (day_df.index.month == j.month) & (day_df.index.year == j.year)
           rep_days = np.sum(date_indexer)
           day_df.loc[date_indexer,string] = [i] * rep_days
    return(day_df)

def clean_cycle_merge(merged_df, string, default = 0):
    new_df = merged_df[merged_df[string] != default]
    return(new_df)
    
def normalize_probs(probs_df):
    new_probs = probs_df.drop('pred_date', axis = 1)
    sum_row = new_probs.sum(axis = 1)
    norm_probs = new_probs.div(sum_row, axis = 0)
    return(norm_probs)

def expected_downside_risk_v1(merged_df,norm_probs,thresh=0,power=2):
    thresh_df = merged_df - thresh
    thresh_df['Cycle'] += thresh
    index_var = merged_df['Cycle'].unique()
    index_ticker = merged_df.columns.values[:-1]
    downside_df = pd.DataFrame(index=index_ticker, columns=index_var)    
    prob_vector = pd.np.array(norm_probs.tail(1))    
    
    for i in index_var:
        for j in index_ticker:
            initial_vector=thresh_df.loc[thresh_df['Cycle'] == i][j]
            downside_df.ix[j,i]=(initial_vector[ initial_vector < 0 ] ** 2).mean()
    
    downside_metrics = 0
    for i in index_var:
        downside_metrics += downside_df.loc[:,i] * prob_vector[:,i-1]
    
    return(downside_metrics)

def expected_omega_v1(merged_df,norm_probs,thresh=0):
    thresh_df = merged_df - thresh
    thresh_df['Cycle'] += thresh
    index_var = merged_df['Cycle'].unique()
    index_ticker = merged_df.columns.values[:-1]
    upside_omega_df = pd.DataFrame(index=index_ticker, columns=index_var)
    downside_omega_df = pd.DataFrame(index=index_ticker, columns=index_var)
    omega_df = pd.DataFrame(index=index_ticker, columns=index_var)
    prob_vector = pd.np.array(norm_probs.tail(1))    
    
    for i in index_var:
        for j in index_ticker:
            initial_vector=thresh_df.loc[thresh_df['Cycle'] == i][j]
            upside_omega_df.ix[j,i]=((initial_vector[ initial_vector > 0 ])).sum()
            downside_omega_df.ix[j,i]=(-1) * ((initial_vector[ initial_vector < 0 ])).sum()

            if downside_omega_df.ix[j,i] != 0: 
                omega_df.ix[j,i]= upside_omega_df.ix[j,i] / downside_omega_df.ix[j,i]
            else:
                omega_df.ix[j,i]= np.nan
            
    upside_omega_metrics = 0
    downside_omega_metrics = 0
    omega_metrics = 0
    for i in index_var:
        upside_omega_metrics    += upside_omega_df.loc[:,i] * prob_vector[:,i-1]
        downside_omega_metrics  += downside_omega_df.loc[:,i] * prob_vector[:,i-1]
        omega_metrics           += omega_df.loc[:,i] * prob_vector[:,i-1]
        
    return(upside_omega_metrics,downside_omega_metrics,omega_metrics)

def expected_absolute_downside_risk_v1(merged_df,norm_probs,thresh=0,power=2):
    thresh_df = merged_df - thresh
    thresh_df['Cycle'] += thresh
    index_var = merged_df['Cycle'].unique()
    index_ticker = merged_df.columns.values[:-1]
    abs_downside_df = pd.DataFrame(index=index_ticker, columns=index_var)    
    prob_vector = pd.np.array(norm_probs.tail(1))    
    
    for i in index_var:
        for j in index_ticker:
            initial_vector=thresh_df.loc[thresh_df['Cycle'] == i][j]
            abs_downside_df.ix[j,i]=(abs(initial_vector[ initial_vector < 0 ])).mean()
    
    abs_downside_metrics = 0
    for i in index_var:
        abs_downside_metrics += abs_downside_df.loc[:,i] * prob_vector[:,i-1]
    
    return(abs_downside_metrics)

def expected_returns_prob_v1(merged_df,norm_probs):
    summary_df = merged_df.groupby('Cycle', as_index = True).mean().T
    prob_vector = pd.np.array(norm_probs.tail(1))
    expected_return = summary_df.dot(prob_vector.T)
    return(expected_return)

def covariance_matrix_prob_v1(merged_df,norm_probs):
    total_cov = merged_df.groupby('Cycle', as_index=True).cov()
    index_var = merged_df['Cycle'].unique()
    prob_vector = pd.np.array(norm_probs.tail(1))

    cov_matrix = 0
    for i in index_var:
        cov_matrix += total_cov.loc[i,:] * prob_vector[:,i-1]
    return(cov_matrix, total_cov)

def month_aggregation(dayly_excess):
    how_dict={}
    for i in dayly_excess.columns.values:
        if i == 'Cycle':
            how_dict[i] = 'last'
        else:
            how_dict[i] = 'sum'
    month_excess_returns = dayly_excess.resample('M',how=how_dict)
    
    return(month_excess_returns)

clean_obs = preprocess_data(observations)
clean_cpi = preprocess_data(cpi_timeseries)
clean_cycle = preprocess_data(cycle)

returns_obs = returns_dataframe(clean_obs,index_type)
returns_obs = returns_obs.dropna()


#TO DO Create the excess returns
dayly_cpi_return = dayly_cpi_return(clean_cpi)

#Finish the excess returns
excess_returns_obs = excess_returns(dayly_cpi_return,returns_obs)

index_cycle = merge_cycle_obs(excess_returns_obs,clean_cycle,'Cycle')
clean_index_cycle = clean_cycle_merge(index_cycle,'Cycle')
normal_probs = normalize_probs(hist_probs)
Downside = expected_downside_risk_v1(clean_index_cycle, normal_probs)
Abs_Downside = expected_absolute_downside_risk_v1(clean_index_cycle, normal_probs)
ERs = expected_returns_prob_v1(clean_index_cycle, normal_probs)
(Cov_matrix,total_cov) = covariance_matrix_prob_v1(clean_index_cycle, normal_probs)
(UpOmega,DownOmega,Omega) = expected_omega_v1(clean_index_cycle, normal_probs,thresh = 0.001) 