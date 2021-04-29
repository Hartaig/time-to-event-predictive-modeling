import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
import sklearn
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


class TimeToEvent(object):
    """
    This class creates an TimeToEvent object that contains the initial data set, features, time column, event column,
    option to disable feature selection, and number of features to choose if feature selection is enabled.
    """

    def __init__(self, dat, features, time, event, enable_fs, top_n=[]):
        """

        Parameters
        ----------
        dat: dataframe
            Data set that contains all features, time columns, and event columns. Must be all numerics. Categorical features
            must be one-hot encoded, ordinal categorical features must be numerically encoded, continuous features can
            remain as is.

        features: list
            List of features to training model on.

        time: list
            Column name that indicates which column in dat is for time to event.

        event: list
            Column name that indicates which column in dat is for then event occurring or not. Binary 0/1 indicator. 0
            implies right-censoring.

        enable_fs: boolean
            True/false indicating whether to perform features selection

        top_n: int
            If feature selection is enabled, the number of top features to include in model fitting.

        """
        self.dat = dat
        self.features = features
        self.time = time
        self.event = event
        self.enable_fs = enable_fs
        self.top_n = top_n
