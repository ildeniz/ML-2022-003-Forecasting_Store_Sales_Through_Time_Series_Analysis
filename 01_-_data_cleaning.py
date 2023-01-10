# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:22:17 2023

@author: ildeniz
"""

#%% 
# loading initial packages and libraries
import os
import pandas as pd
# import numpy as np

# Setting the working directory
path = os.getcwd()

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
os.chdir(dir_path)

#%%
df_store = pd.read_csv('Data\store.csv')

""" WARNING!
`\t` part works as a/an 'tab/indent' so this generates a data loading problem. 
Use '\\' or '/t' instead of '\t'
"""

df_rossmann = pd.read_csv('Data/train.csv')
# we don't need to load 'test.csv' since we won't participate into the competition

# A warning message appeared while loading 'train.csv'
# =============================================================================
# DtypeWarning: Columns (7) have mixed types. Specify dtype option on import or set low_memory=False.
# =============================================================================

# get the name of column 7
df_rossmann.columns[7]

# elements of 'StateHoliday'
df_rossmann['StateHoliday'].unique()

# 'train.csv' is consist of two types of zeros, one is in string type
# and the other one is in integer type. Since 0 stands for 'no holiday', and the 
# other entries are 'str' type, it is better to change 'int' types into 'str'
# to deal with the confusion.

# #%%time

df_rossmann['StateHoliday'] = [str(int(x)) if isinstance(x, int) else x for x in df_rossmann['StateHoliday']]

df_rossmann['StateHoliday'].unique()

df_rossmann.isnull().sum()
# no missing values in 'rossmann.csv'

df_store.isnull().sum()

"""
TODO: extract the part below and turn it into a function
"""
total = df_store.isnull().sum().sort_values(ascending=False)
percent = (df_store.isnull().sum())/(df_store.isnull().count().sort_values(ascending=False))*100
missing_values=pd.concat([total,percent],keys=['Total','Percent'],axis=1)
missing_values

#                            Total    Percent
# Promo2SinceWeek              544  48.789238
# Promo2SinceYear              544  48.789238
# PromoInterval                544  48.789238
# CompetitionOpenSinceMonth    354  31.748879
# CompetitionOpenSinceYear     354  31.748879
# CompetitionDistance            3   0.269058

df_store.info()

df_store['CompetitionDistance'].skew()
# 2.9292856455312055

import seaborn as sns

sns.kdeplot(df_store['CompetitionDistance'], shade=True)
# sns.histplot(df_store['CompetitionDistance'])

# since 'CompetitionDistance' is highly skewed, it would be better to fill 
# the missing values with the median value
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace = True)

# merge store data with the main data set
df_rossmann = df_rossmann.merge(df_store,how='left',on='Store')
df_rossmann.info()

# Get list of categorical variables
cat_vars = (df_rossmann.dtypes == 'object')
object_cols = list(cat_vars[cat_vars].index)

"""
'Date'          -> convert to 'datetime'
'StateHoliday'  -> one-hot-encoding
'StoreType'     -> one-hot-encoding
'Assortment'    -> one-hot-encoding
'PromoInterval' -> this column has missing values, I skip this one for now
"""

# 'Date' convert to 'datetime'
df_rossmann['date_parsed'] = df_rossmann['Date'].copy()
df_rossmann['date_parsed'] = pd.to_datetime(df_rossmann['date_parsed'], format='%Y-%m-%d')
df_rossmann['date_parsed'].dtypes

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: df_rossmann[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

# looks like it worths to apply one-hot-encoding on 'Assortment', 'StoreType', and 'StateHoliday'
# [('Assortment', 3),
#  ('PromoInterval', 3),
#  ('StateHoliday', 4),
#  ('StoreType', 4),
#  ('Date', 942)]

# one-hot encoding the categorical columns
one_hot_encoded_data = pd.get_dummies(df_rossmann[['Assortment', 'StateHoliday', 'StoreType']], columns = ['Assortment', 'StateHoliday', 'StoreType'])

# excluding catergorical columns
num_df_rossmann = df_rossmann.drop(['Assortment', 'StateHoliday', 'StoreType'], axis=1)

# merging new categorical columns with existing numerical columns
df_rossmann = pd.concat([num_df_rossmann, one_hot_encoded_data], axis=1)

# to shrink the '*.csv' file, get rid of the unused columns
df_rossmann = df_rossmann.drop(['Date',
                                'Promo2SinceWeek',
                                'Promo2SinceYear',
                                'PromoInterval',
                                'CompetitionOpenSinceMonth',
                                'CompetitionOpenSinceYear'],axis=1)

# saving the cleaned data set as a '*.csv' file
df_rossmann.to_csv('Data/rossmann_cleaned.csv', index=False)
