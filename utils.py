# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:46:01 2023

@author: ildeniz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def seasonal_decompose (df, title, period=7, model='additive'):
    '''
    model{“additive”, “multiplicative”}, optional
    '''

    import statsmodels.api as sm
    
    decomposition = sm.tsa.seasonal_decompose(df, model=model, period=period)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    fig = decomposition.plot()
    fig.suptitle(title)
    fig.set_size_inches(14, 7) 
    plt.show()
    
    return trend, seasonal, residual

'''
obtained from: https://github.com/Kaggle/learntools/blob/master/learntools/time_series/utils.py
'''
def make_mi_scores(X, y, discrete_features, target_is = "real_valued"):
    '''
    'target_is' = {"real_valued", "categorical"} default "real_valued" 
        parameter has two argument, whether target (y) is 'real_valued' or 'categorical'
    '''
    from sklearn.feature_selection import (mutual_info_regression, 
                                           mutual_info_classif)
    
    if target_is == "real_valued":
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    elif target_is == "categorical":
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
        
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)