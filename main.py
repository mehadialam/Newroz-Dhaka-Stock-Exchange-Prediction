﻿# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
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
 
 
def read_data(path, start_year=2008, end_year=2020):
    return pd.concat([pd.read_json(path + '/prices_%d.json' % x)
                     for x in range(start_year, end_year)])
 
 
df = read_data('/content/drive/My Drive/DseDataSet')
 
 
def get_all_ticker(dataframe=df, start_date='2019-01-01',
                   end_date='2019-12-31'):
    '''
    Return a list for all trading code of Stock data
    '''
 
    return dataframe.loc[(dataframe['date'] >= start_date)
                         & (dataframe['date']
                         <= end_date)].trading_code.unique()
 
 
def preprocess(ticker, dataframe):
    ''' 
    If any stoke data face value other than 10TK then make it 10 taka. 
    this function will return stock_data = face value 10taka stock data.
    '''
 
    stock_data = dataframe.loc[dataframe['trading_code'] == ticker]
    stock_data = stock_data.sort_values(by='date')
    stock_data = stock_data.drop_duplicates()
    stock_data = stock_data.reset_index(drop=True)
    stock_data = stock_data.rename(columns={'closing_price': 'y',
                                   'date': 'ds'})
    stock_data.loc[stock_data['y'] == 0, 'y'] = stock_data.loc[:,
            'yesterdays_closing_price'].shift(periods=-1)
 
    if ticker in ['1STBSRS', '1STICB', '2NDICB', '3RDICB', '4THICB', '5THICB',
       '6THICB', '7THICB', '8THICB', 'ABBANK', 'AGRANINS', 'ALLTEX',
       'AMCL(PRAN)', 'ANLIMAYARN', 'ANWARGALV', 'APEXADELFT', 'APEXFOODS',
       'APEXSPINN', 'APEXTANRY', 'ARAMITCEM', 'ASIAPACINS', 'AZIZPIPES',
       'BANGAS', 'BANKASIA', 'BDAUTOCA', 'BDLAMPS', 'BDTHAI', 'BIFC',
       'BRACBANK', 'BXSYNTH', 'CENTRALINS', 'CITYBANK', 'CONTININS',
       'CVOPRL', 'DBH', 'DELTALIFE', 'DELTASPINN', 'DESCO', 'DSHGARME',
       'DULAMIACOT', 'DUTCHBANGL', 'EASTERNINS', 'EASTLAND', 'ECABLES',
       'EHL', 'FLEASEINT', 'FUWANGCER', 'GEMINISEA', 'GLOBALINS',
       'HEIDELBCEM', 'HRTEX', 'IBNSINA', 'ICB1STNRB', 'ICB2NDNRB',
       'ICBAMCL1ST', 'ICBISLAMIC', 'IDLC', 'IFIC', 'ILFSL', 'IMAMBUTTON',
       'IPDC', 'ISLAMIBANK', 'ISLAMICFIN', 'JUTESPINN', 'KAY&QUE',
       'KOHINOOR', 'LAFSURCEML', 'LIBRAINFU', 'MERCANBANK', 'MERCINS',
       'MIDASFIN', 'MITHUNKNIT', 'MODERNDYE', 'MONNOCERA', 'MONNOSTAF',
       'MTBL', 'NATLIFEINS', 'NITOLINS', 'NORTHRNINS', 'NPOLYMAR', 'NTC',
       'NTLTUBES', 'OLYMPIC', 'ONEBANKLTD', 'ORIONINFU', 'PARAMOUNT',
       'PHARMAID', 'PHOENIXFIN', 'PIONEERINS', 'POWERGRID', 'PRAGATIINS',
       'PRAGATILIF', 'PREMIERLEA', 'PRIMETEX', 'PROGRESLIF', 'PURABIGEN',
       'RAHIMAFOOD', 'RAHIMTEXT', 'RELIANCINS', 'RENATA', 'RENWICKJA',
       'RUPALIBANK', 'SAFKOSPINN', 'SALAMCRST', 'SAMATALETH', 'SAMORITA',
       'SAVAREFR', 'SINGERBD', 'SONALIANSH', 'SONARBAINS', 'SONARGAON',
       'SOUTHEASTB', 'SQURPHARMA', 'STANCERAM', 'STANDARINS',
       'STANDBANKL', 'STYLECRAFT', 'TAKAFULINS', 'TALLUSPIN', 'TITASGAS',
       'TRUSTBANK', 'ULC', 'UNITEDINS', 'USMANIAGL', 'BAYLEASING',
       'BSRMSTEEL', 'ICBAMCL2ND', 'ISLAMIINS', 'NHFIL', 'REPUBLIC',
       'RUPALILIFE', 'DHAKAINS', 'PROVATIINS', 'FASFIN']:
        stock_data_slice = stock_data.loc[stock_data['ds']
                < '2011-12-01']
 
        stock_data_slice[[
            'y',
            'high',
            'low',
            'last_traded_price',
            'opening_price',
            'yesterdays_closing_price',
            ]] = stock_data_slice.loc[:, [
            'y',
            'high',
            'low',
            'last_traded_price',
            'opening_price',
            'yesterdays_closing_price',
            ]] / 10
 
        stock_data_slice['volume'] = stock_data_slice.loc[:, 'volume'] \
            * 10
        stock_data = pd.concat([stock_data_slice,
                               stock_data.loc[stock_data['ds']
                               >= '2011-12-01']])
    return stock_data
 
 
def boxCoxTransformation(ticker, dataframe):
    '''
    this function will return stock_data_box = box normalization of stock data, 
    box_lam = lamda value for box.
    '''
 
    stock_data = preprocess(ticker=ticker, dataframe=dataframe)
    stock_data_box = stock_data.loc[:, ['ds', 'y']]
    (stock_data_box['y'], box_lam) = boxcox(stock_data_box['y'])
    return (stock_data_box, box_lam)
 
 
def model(
    ticker,
    dataframe=df,
    start_date='2008-01-01',
    end_date='2018-12-31',
    growth='linear',
    changepoints=None,
    n_changepoints=25,
    changepoint_range=0.8,
    yearly_seasonality='auto',
    weekly_seasonality=False,
    daily_seasonality=False,
    holidays=None,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.8,
    uncertainty_samples=1000,
    stan_backend=None,
    monthly_seasonality_order=5,
    ):
    ''' 
    Return dictionary of prediction and actual value
    '''
 
    (stock_data_box, box_lam) = boxCoxTransformation(ticker=ticker,
            dataframe=dataframe)
    stock_data_actual = stock_data_box.copy()
    stock_data_actual['y'] = stock_data_actual['y'].apply(lambda x: \
            inv_boxcox(x, box_lam))
 
    m = Prophet(
        growth=growth,
        changepoints=changepoints,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        holidays=holidays,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples,
        stan_backend=stan_backend,
        )
 
    m.add_seasonality(name='monthly', period=30.5,
                      fourier_order=monthly_seasonality_order)
    return m
 
 
def model_forecast(
    ticker,
    dataframe=df,
    start_date='2008-01-01',
    end_date='2018-12-31',
    growth='linear',
    changepoints=None,
    n_changepoints=25,
    changepoint_range=0.8,
    yearly_seasonality='auto',
    weekly_seasonality=False,
    daily_seasonality=False,
    holidays=None,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.8,
    uncertainty_samples=1000,
    stan_backend=None,
    monthly_seasonality_order=5,
    futute_periods=370,
    future_freq='D',
    ):
 
    (stock_data_box, box_lam) = boxCoxTransformation(ticker, dataframe)
    stock_data_actual = stock_data_box.copy()
    stock_data_actual['y'] = stock_data_actual['y'].apply(lambda x: \
            inv_boxcox(x, box_lam))
 
    m = model(
        ticker=ticker,
        dataframe=dataframe,
        start_date=start_date,
        end_date=end_date,
        growth=growth,
        changepoints=changepoints,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        holidays=holidays,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples,
        stan_backend=stan_backend,
        monthly_seasonality_order=monthly_seasonality_order,
        )
    m.fit(stock_data_box.loc[(stock_data_box['ds'] >= start_date)
          & (stock_data_box['ds'] <= end_date)])
    future = m.make_future_dataframe(periods=futute_periods,
            freq=future_freq)
    forecast = m.predict(future)
    forecast[['yhat', 'yhat_upper', 'yhat_lower']] = forecast[['yhat',
            'yhat_upper', 'yhat_lower']].apply(lambda x: inv_boxcox(x,
            box_lam))
    #data = {'prediction': forecast[['ds', 'yhat']],
    #        'actual': stock_data_actual[['ds', 'y'
    #       ]].loc[(stock_data_actual['ds'] >= start_date)
    #        & (stock_data_actual['ds'] <= end_date)]}
    data = stock_data_actual[['ds', 'y']].loc[(stock_data_actual['ds'] >= start_date) & (stock_data_actual['ds'] <= end_date)].merge(forecast[['ds', 'yhat']], on ='ds', how = 'outer')
    return data
 
 
def visualstockdata(stock_data):
    fig = go.Figure()
 
    # Create and style traces
 
    fig.add_trace(go.Scatter(x=stock_data['ds'], y=stock_data['y'],
                  name='Equity'))
    fig.show()
 
 
def find_best_params(
    ticker,
    dataframe=df,
    start_date='2008-01-01',
    end_date='2018-12-31',
    horizon='361 days',
    initial=None,
    period=None,
    parallel=None,
    cutoffs=None,
    base_error='rmse',
    **param_dict
    ):
 
    (stock_data_box, box_lam) = boxCoxTransformation(ticker, dataframe)
    param_iter = itertools.product(*param_dict.values())
    params = []
    for param in param_iter:
        params.append(param)
    params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
 
    metrics = [
        'horizon',
        'mse',
        'rmse',
        'mae',
        'mape',
        'mdape',
        'coverage',
        'params',
        ]
    results = []
    count_combination = 1
    for param in params_df.values:
        print ('Hyper-parameter combination set:', count_combination)
        param_dict = dict(zip(params_df.keys(), param))
        try:
            monthly_seasonality_order = \
                param_dict['monthly_seasonality_order']
            param_dict.pop('monthly_seasonality_order')
        except:
            monthly_seasonality_order = 5
        m = Prophet(**param_dict)
        m.add_seasonality(name='monthly', period=30.5,
                          fourier_order=monthly_seasonality_order)
        m.fit(stock_data_box.loc[(stock_data_box['ds'] >= start_date)
              & (stock_data_box['ds'] <= end_date)])
 
        df_cv = cross_validation(
            m,
            horizon=horizon,
            initial=initial,
            period=period,
            parallel=parallel,
            cutoffs=cutoffs,
            )
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['params'] = str(param_dict)
        df_p = df_p.loc[:, metrics]
        results.append(df_p)
        count_combination = count_combination + 1
    results_df = pd.concat(results).reset_index(drop=True)
    best_param = results_df.loc[results_df[base_error]
                                == min(results_df[base_error]),
                                ['params']]
    return eval(best_param.values[0][0])
 
 
def predict(
    ticker,
    dataframe=df,
    start_date='2008-01-01',
    end_date='2018-12-31',
    horizon='361 days',
    initial=None,
    period=None,
    parallel=None,
    cutoffs=None,
    prediction_start_date='2019-01-01',
    prediction_end_date='2019-12-31',
    base_error='rmse',
    **param_dict
    ):
 
    best_params = find_best_params(
        ticker=ticker,
        dataframe=dataframe,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        initial=initial,
        period=period,
        parallel=parallel,
        cutoffs=cutoffs,
        base_error=base_error,
        **param_dict
        )
    data = model_forecast(ticker=ticker, dataframe=dataframe,
                          start_date='2008-01-01', end_date='2018-12-31'
                          , **best_params)
    prediction = data['prediction']
    prediction.rename(columns={'yhat': 'Predicted_Closing_Price',
                      'ds': 'Date'}, inplace=True)
    return prediction.loc[(prediction['Date'] >= prediction_start_date)
                          & (prediction['Date'] <= prediction_end_date)]