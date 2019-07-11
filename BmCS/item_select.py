"""
Module for ensemble feature transformation using sklearn
"""

from sklearn.base import BaseEstimator, TransformerMixin
import re
import numpy as np

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Class for ensemble feature transformation

    For data grouped by feature, select subset of data at a provided key.
    This is necessary when providing sklearn predict or predict_proba
    method with data structure with multiple columns or keys.
    """

    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.column]

    def get_feature_names(self):
        return df[self.column].columns.tolist()
