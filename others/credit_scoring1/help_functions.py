import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, year=True, month=True, day_of_month=True, day_of_week=True):
        self.year = year
        self.month = month
        self.day_of_month = day_of_month
        self.day_of_week = day_of_week
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        ## Checking dtype of column
        X = X.squeeze() ### From DataFrame to Series
        if not pd.api.types.is_datetime64_ns_dtype(X):
            raise(Exception('!!! Incorrect dtype of column. Its dtype must be datetime64_ns !!!'))
    
        encoded_X = pd.DataFrame().copy()
    
        ## Extracting "year"
        if self.year:
            encoded_X['year'] = X.dt.year
    
        ## Extracting "month"
        if self.month:
            encoded_X['month'] = X.dt.month
            encoded_X['norm_month'] = 2 * np.pi * encoded_X.month / encoded_X.month.max()
            encoded_X['sin_month'] = np.sin(encoded_X.norm_month)
            encoded_X['cos_month'] = np.cos(encoded_X.norm_month)
            encoded_X.drop(columns=['month','norm_month'], inplace=True)
    
        ## Extracting "day_of_month"
        if self.day_of_month:
            encoded_X['day_of_month'] = X.dt.day
            encoded_X['norm_day_of_month'] = 2 * np.pi * encoded_X.day_of_month / encoded_X.day_of_month.max()
            encoded_X['sin_day_of_month'] = np.sin(encoded_X.norm_day_of_month)
            encoded_X['cos_day_of_month'] = np.cos(encoded_X.norm_day_of_month)
            encoded_X.drop(columns=['day_of_month','norm_day_of_month'], inplace=True)
    
        ## Extracting "day_of_week"
        if self.day_of_week:
            encoded_X['day_of_week'] = X.apply(lambda date: date.weekday())
            encoded_X['norm_day_of_week'] = 2 * np.pi * encoded_X.day_of_week / encoded_X.day_of_week.max()
            encoded_X['sin_day_of_week'] = np.sin(encoded_X.norm_day_of_week)
            encoded_X['cos_day_of_week'] = np.cos(encoded_X.norm_day_of_week)
            encoded_X.drop(columns=['day_of_week','norm_day_of_week'], inplace=True)
        
        return encoded_X

############################################################################################################################################

