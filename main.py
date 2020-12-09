import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

import itertools


def read_data(path , start_year=2008, end_year=2020):
    return pd.concat([pd.read_json(path + '/prices_%d.json' %x) for x in range(start_year, end_year)])


def get_all_ticker(dataframe=read_data('/content/drive/My Drive/DseDataSet'), start_date='2019-01-01', end_date='2019-12-31'):
    '''
    Return a list for all trading code of Stock data
    '''
    return dataframe.loc[(dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)].trading_code.unique()


def preprocess(ticker, dataframe):
    ''' 
    If any stoke data face value other than 10TK then make it 10 taka. 
    this function will return stock_data = face value 10taka stock data.
    '''
   
    stock_data = dataframe.loc[dataframe['trading_code'] == ticker]
    stock_data = stock_data.sort_values(by='date')
    stock_data = stock_data.drop_duplicates()
    stock_data = stock_data.reset_index(drop=True)
    stock_data = stock_data.rename(columns={'closing_price': 'y', 'date': 'ds'})
    stock_data.loc[stock_data['y'] == 0, 'y'] = stock_data.loc[:, 
                                                               'yesterdays_closing_price'].shift(periods = -1)
    
    if ticker in ['SQURPHARMA', 'SINGERBD']:
        stock_data_slice = stock_data.loc[stock_data['ds'] < '2011-12-01']
        stock_data_slice[[
                          'y', 'high', 
                          'low', 'last_traded_price', 
                          'opening_price', 
                          'yesterdays_closing_price']] = stock_data_slice.loc[:, 
                                                                              [ 'y', 'high', 'low', 
                                                                                'last_traded_price', 
                                                                                'opening_price', 
                                                                                'yesterdays_closing_price']]/10
        
        stock_data_slice['volume']= stock_data_slice.loc[:,'volume']*10
        stock_data = pd.concat([stock_data_slice, stock_data.loc[stock_data['ds'] >= '2011-12-01']])
  
    return stock_data


def boxCoxTransformation(ticker, dataframe):
    '''
    this function will return stock_data_box = box normalization of stock data, 
    box_lam = lamda value for box.
    '''
    stock_data = preprocess(ticker, dataframe)      
    stock_data_box = stock_data.loc[:, ['ds', 'y']]
    stock_data_box['y'], box_lam = boxcox(stock_data_box['y'])
    return stock_data_box, box_lam


def model(ticker, dataframe=read_data('/content/drive/My Drive/DseDataSet'),  
             start_date='2008-01-01', end_date='2018-12-31', growth='linear', changepoints=None, n_changepoints=25, 
             changepoint_range=0.8, yearly_seasonality='auto', weekly_seasonality=False, 
             daily_seasonality=False, holidays=None, seasonality_mode='additive', 
             seasonality_prior_scale=10.0, holidays_prior_scale=10.0, 
             changepoint_prior_scale=0.05, mcmc_samples=0, interval_width=0.8, 
             uncertainty_samples=1000, stan_backend=None):
    ''' 
    Return dictionary of prediction and actual value
    '''

    stock_data_box, box_lam = boxCoxTransformation(ticker, dataframe)
    stock_data_actual = stock_data_box.copy()
    stock_data_actual['y'] = stock_data_actual['y'].apply(lambda x: inv_boxcox(x, box_lam))
  
    m = Prophet(growth=growth, changepoints=changepoints, n_changepoints=n_changepoints, 
                changepoint_range=changepoint_range, yearly_seasonality=yearly_seasonality, 
                weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality, holidays=holidays, 
                seasonality_mode=seasonality_mode, seasonality_prior_scale=seasonality_prior_scale, 
                holidays_prior_scale=holidays_prior_scale, changepoint_prior_scale=changepoint_prior_scale, 
                mcmc_samples=mcmc_samples, interval_width=interval_width, uncertainty_samples=uncertainty_samples, 
                stan_backend=stan_backend) 

    m.add_seasonality(name='monthly', period=30.5, fourier_order=15)

    m.fit(stock_data_box.loc[(stock_data_box['ds'] >= start_date) & (stock_data_box['ds'] <= end_date)])

    future = m.make_future_dataframe(periods=370, freq='D')

    forecast = m.predict(future)

    forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, box_lam))
  
    data = {'prediction': forecast[['ds','yhat']],
            'actual': stock_data_actual[['ds', 'y']].loc[(stock_data_actual['ds'] >= start_date) & (stock_data_actual['ds'] <= end_date)]}
    return data


def visualstockdata(stock_data):
    fig = go.Figure()
    #Create and style traces
    fig.add_trace(go.Scatter(x = stock_data['ds'], y = stock_data['y'], name= 'Equity',))
    fig.show()
