# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:08:50 2023

@author: ildeniz
"""

#%% 
# Setting the working directory
import os

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
os.chdir(dir_path)

#%%
# loading initial packages and libraries
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# loading the data set
df = pd.read_csv('Data/rossmann_cleaned.csv', 
                    parse_dates=['Date']
                   )

df['date_parsed'] = df['Date'].copy()
#df = df.set_index('date_parsed').to_period('D')

# choosing the relevant columns
df.columns
df_model = df[['Store', 'DayOfWeek', 'Date', 'Sales', 'Promo', 'date_parsed', 'Customers',]]

df_model = df_model.set_index('date_parsed').to_period('D')
df_model = df_model.set_index(['Store'], append=True)
df_model = df_model.groupby('Date').mean()[['Sales','Customers']]


# State specific/regional holidays (determined through EDA)
nationWideHolidayList = ['2013-01-01','2013-03-29','2013-04-01','2013-05-01','2013-05-09','2013-05-20','2013-05-30','2013-10-03','2013-11-01','2013-12-25','2013-12-26',
                         '2014-01-01','2014-04-18','2014-04-21','2014-05-01','2014-05-29','2014-06-09','2014-06-19','2014-10-03','2014-11-01','2014-12-25','2014-12-26',
                         '2015-01-01','2015-04-03','2015-04-06','2015-05-01','2015-05-14','2015-05-25','2015-06-04']
df['National_Holiday'] = 0
for date in range(len(nationWideHolidayList)):
    df.loc[df['Date'] == nationWideHolidayList[date], 'National_Holiday'] = 1


# National holidays when >500 stores are closed in Germany wide.
df_model['is_holiday'] = df.groupby('Date').max()['National_Holiday']#.to_frame() #.iloc[1:]
df_model['hol_next_day'] = df_model['is_holiday'].shift(1).fillna(0)

# Promotion days
df_model['promo_days'] = df.groupby('Date').max()['Promo']

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression

model_1 = [LinearRegression(fit_intercept=False),
           ElasticNet(fit_intercept=False),
           Lasso(fit_intercept=False),
           Ridge(fit_intercept=False),
           ]

# Model 2 (cycles)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# defining the error measure function
def root_mean_square_percentage_error(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))
    return loss

model_2 = [XGBRegressor(),
           RandomForestRegressor(),
           ExtraTreesRegressor(),
           KNeighborsRegressor(),
           MLPRegressor(),
           ]


X = df_model.copy()
y = X.pop('Sales').to_frame()

# deterministic process features
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=y['Sales'].index,
    constant=True,
    order=1,
    seasonal=True,
    drop=True,
)

X_All = X.copy().join(dp.in_sample())

# model specific features
X_1 = X_All[['const', 'trend']]

from sklearn.model_selection import train_test_split
X1_train, X1_test, y_train, y_test = train_test_split(X_1, y, test_size=60, shuffle=False)

X_2 = X_All.loc[:, ~X_All.columns.isin(['const', 'trend'])]
X2_train, X2_test, y_train, y_test = train_test_split(X_2, y, test_size=60, shuffle=False)

# define fitting and predicting function
def fit_and_predict(model_1, model_2, X1_train, X2_train, X1_test, X2_test, y_test, y_train):
    model_1.fit(X1_train, y_train)
    model_1_fit = pd.DataFrame(
        model_1.predict(X1_train),
        index=X1_train.index, columns=y_train.columns,
    )

    y_resid_1 = y_train - model_1_fit

    from utils import make_lags

    lags = make_lags(y_resid_1, 9).fillna(0)
    X2_train = pd.concat([lags ,X2_train], axis=1, copy=False)

    model_2.fit(X2_train, y_resid_1)

    # predict
    m1_pred = pd.DataFrame(
        model_1.predict(X1_test),
        index=X1_test.index, columns=y_test.columns,
    )

    y_res_pred = y_test - m1_pred

    lags_test = make_lags(y_res_pred, 9).fillna(0)
    X2_test = pd.concat([lags_test ,X2_test], axis=1, copy=False)

    m2_pred = pd.DataFrame(
        model_2.predict(X2_test),
        index=X2_test.index, columns=y_res_pred.columns,
    )

    y_pred = m1_pred + m2_pred
    
    rmspe_valid = root_mean_square_percentage_error(y_test['Sales'], y_pred['Sales'])
        
    print(model_1, 'and', model_2, f'Validation RMSPE: {rmspe_valid:.4f}')
    return y_pred

# obtaining the error measumenets of hybrid model pairs with default parameters
for model in model_1:
    m_1 = model
    for model in model_2:
        m_2 = model
        fit_and_predict(m_1, m_2, X1_train, X2_train, X1_test, X2_test, y_test, y_train)
# =============================================================================
# Results
# LinearRegression() and XGBRegressor()             Validation RMSPE: 0.1774
# LinearRegression() and RandomForestRegressor()    Validation RMSPE: 0.2089
# LinearRegression() and ExtraTreesRegressor()      Validation RMSPE: 0.2071
# LinearRegression() and KNeighborsRegressor()      Validation RMSPE: 4.0648
# LinearRegression() and MLPRegressor()             Validation RMSPE: 2.1703
# ElasticNet() and XGBRegressor()                   Validation RMSPE: 1.2337
# ElasticNet() and RandomForestRegressor()          Validation RMSPE: 1.6776
# ElasticNet() and ExtraTreesRegressor()            Validation RMSPE: 1.7922
# ElasticNet() and KNeighborsRegressor()            Validation RMSPE: 3.8026
# ElasticNet() and MLPRegressor()                   Validation RMSPE: 2.6762
# Lasso() and XGBRegressor()                        Validation RMSPE: 0.2014
# Lasso() and RandomForestRegressor()               Validation RMSPE: 0.2087
# Lasso() and ExtraTreesRegressor()                 Validation RMSPE: 0.2121
# Lasso() and KNeighborsRegressor()                 Validation RMSPE: 4.0664
# Lasso() and MLPRegressor()                        Validation RMSPE: 2.0207
# Ridge() and XGBRegressor()                        Validation RMSPE: 0.2575
# Ridge() and RandomForestRegressor()               Validation RMSPE: 0.2181
# Ridge() and ExtraTreesRegressor()                 Validation RMSPE: 0.2258
# Ridge() and KNeighborsRegressor()                 Validation RMSPE: 4.0958
# Ridge() and MLPRegressor()                        Validation RMSPE: 2.2222
# =============================================================================

# LinearRegression() and XGBRegressor() do the best job with the default parameters, it worts to tune these models

# producing the comparison plot
y_pred = fit_and_predict(LinearRegression(), XGBRegressor(), X1_train, X2_train, X1_test, X2_test, y_test, y_train)

y_test['Sales'].plot(legend=True, label='Actual',figsize=(15,5)).set_title('Actual vs Prediction Comparison')
y_pred['Sales'].plot(legend=True, label='Prediction', alpha=0.7);

# 1.1- tune models GridsearchCV

# 2- SARIMAX [WIP]

# 3- FB Prophet [WIP]