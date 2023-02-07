# Data Science Store Sales Forecasting Through TSA: Project Overview

## Code and Resources Used 
**Python Version:** 3.9  
**Packages:** pandas(1.4.4), numpy(1.21.5), seaborn(0.11.2), matplotlib(3.5.2), scipy(1.9.1), statsmodels(0.13.2), sklearn(1.0.2). xgboost(1.7.2)

## Data Cleaning & Feature Engineering
- The `'StateHoliday'` column consisted of two types of *zeros*. Since other entries are already string type, integer type zeroes have been changed into strings to deal with the confusion.
- Considering the skewness, missing values in the `'CompetitionDistance'` column have been filled with median values.
- Categorical features such as `'StateHoliday'`, `'StoreType'`, and `'Assortment'` have been one-hot-encoded.
- Helpful information from the `'train.csv'` and `'store.csv'` data sets have been gathered and saved in the `'rossmann_cleaned.csv'` file.

## EDA
Inspected the existence of time series components, the trend, seasonality, and cyclic behaviour, through an iterative process.
The iterative process is defined as follows;
- First, learn the trend, and subtract it from the series,
- Then learn the seasonality from the detrended series, and subtract the seasons out,
- Finally, learn the cyclic behaviour from the detrended and deseasoned series using lags and leads, and subtract the cycles.

This iterative process is also can be used to make the series stationary. In terms of prediction-making and forecasting, it is essential to make the subjected time series stationary.

During the EDA, I engineered some proper forecasting and prediction-making features, such as holiday and promotion related features and lag features.

![alt text](https://github.com/ildeniz/ML-2022-003-Forecasting_Store_Sales_Through_Time_Series_Analysis/blob/master/residuals_vs_rawdata.png "Residuals and Raw Data Comparison")

## Model Building

## Model performance

## Productionization
