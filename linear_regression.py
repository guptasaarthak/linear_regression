### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data(nobs, nbeta):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
        nbeta (int) the number of covariates in the true simulated
    RETURNS
        simulated_data (dict) contains X, y, and beta vectors.
    """

    # Generate X data (from multivariate normal)
    X = np.random.normal(0, 1, (nobs, nbeta))
    # Generate betas
    beta = np.random.normal(0, 5, nbeta)
    # Add idiosyncratic shock
    eps = np.random.normal(0, 1, nobs)

    # Create y from Y = XB + eps
    y = X.dot(beta) + eps

    # Store results as dictionary
    simulated_data = {"X": X,
                      "y": y,
                      "beta": beta}

    return simulated_data

def compare_models(X, y):
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """



def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass


### END ###
