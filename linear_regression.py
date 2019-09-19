### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os


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
    X[:,0] = np.ones(nobs)
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
    
    regression1 = LinearRegression(fit_intercept = False).fit(X,y)
    regression2 = sm.OLS(y,X).fit()
    
    reg1_coef = np.array(regression1.coef_)
    reg2_coef = np.array(regression2.params)

    beta_coefs_df = pd.DataFrame({"coef_1": reg1_coef, "coef_2": reg2_coef})
    
    return(beta_coefs_df)

def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    data_dir = os.getcwd()
    HospitalData = pd.read_csv(data_dir + "/hospital_charge_sample.csv")
    
    return(HospitalData)


def prepare_data(df):
    
    subset = df.loc[:,("Provider State","Average Medicare Payments")]
    
    X = df["Provider State"]
    y = df["Average Medicare Payments"]
    
    df_out = {"X" : X,
              "y" : y}
    
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    
    return(df_out)


def run_hospital_regression():
    
    hospital_data = load_hospital_data()
    hospital_clean = prepare_data(hospital_data)
    
    hospital_regress = sm.OLS(hospital_clean["y"],hospital_clean["X"]).fit()
    
    hospital_out = np.array(hospital_regress.params)
    
    return(hospital_out)
    
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass


### END ###
