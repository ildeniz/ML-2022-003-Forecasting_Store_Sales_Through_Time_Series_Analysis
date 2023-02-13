# Data Science Store Sales Forecasting Through TSA: Project Overview

## Code and Resources Used 
**Python Version:** 3.9  
**Packages:** pandas(1.4.4), numpy(1.21.5), seaborn(0.11.2), matplotlib(3.5.2), scipy(1.9.1), statsmodels(0.13.2), sklearn(1.0.2). xgboost(1.7.2)

## Data Cleaning & Feature Engineering
- The `'StateHoliday'` column consisted of two types of *zeros*. Since other entries are already string type, integer type zeroes have been changed into strings to deal with the confusion.
- Considering the skewness, missing values in the `'CompetitionDistance'` column have been filled with median values.
- Categorical features such as `'StateHoliday'`, `'StoreType'`, and `'Assortment'` have been one-hot-encoded.
- Helpful information from the `'train.csv'` and `'store.csv'` data sets have been gathered and saved in the `'rossmann_cleaned.csv'` file.
- Additionally engineered features:
  - `'is_holiday'`: This feature shows if >500 stores are closed a particular day or not
  - `'hol_next_day'` : This feature marks the following day of `'is_holiday'`

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
I split the data into train and tests sets with a test size of 60 days.
I tried various combinations of hybrid models and evaluated them using Root Mean Square Percentage Error (RMSPE). 

$$
\textrm{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2},
$$

I chose RMSPE since it was given as the evaluation criteria by Rossmann and also since the errors are squared before they are averaged, the RMSPE gives a relatively high weight to large errors. Hence, RMSPE penalizes really bad predictions.

Models used for hybrid boosting pairs:
- For Trend: Linear, ElasticNet, Lasso, and Ridge Regression
- For Seasonality and Cycles: Extreme Gradient Boosting, Random Forest, Extra Trees, K-Nearest Neighbors, and Multi-layer Perceptron Regression,

## Model performance
Top 5 pairs of hybrid boosting (with default parameter arguments):
- **LinearRegression & XGBRegressor** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: RMSPE = 0.1774
- **Lasso & XGBRegressor** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: RMSPE = 0.2014
- **LinearRegression & ExtraTreesRegressor** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: RMSPE = 0.2071
- **Lasso & RandomForestRegressor** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: RMSPE = 0.2087
- **LinearRegression & RandomForestRegressor** &nbsp;: RMSPE = 0.2089

LinearRegression & XGBRegressor Hybrid Model Results:
![alt text](https://github.com/ildeniz/ildeniz_data_science_portfolio/blob/master/Images/Actual_vs_Prediction_comparison.png "LinearRegression & XGBRegressor Hybrid Model Results")
## Productionization
