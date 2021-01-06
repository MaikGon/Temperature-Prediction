import pandas as pd
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import sktime
# import fbprophet
import pmdarima
import statsmodels.tsa
import pyts
import random
import pickle


def perform_processing(temperature: pd.DataFrame, target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame, serial_number_for_prediction: str) -> float:

    with Path('clf.p').open('rb') as reg_file:  # Don't change the path here
        reg = pickle.load(reg_file)

    sn_number = serial_number_for_prediction

    df_temp = temperature
    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    df_temp.drop(columns=['unit'], inplace=True)

    df_target = target_temperature
    df_target.rename(columns={'value': 'target_temp'}, inplace=True)
    df_target.drop(columns=['unit'], inplace=True)

    df_valve = valve_level
    df_valve.rename(columns={'value': 'valve_level'}, inplace=True)
    df_valve.drop(columns=['unit'], inplace=True)

    df_temp = df_temp[df_temp['serialNumber'] == sn_number]

    df_combined = pd.concat([df_temp, df_target, df_valve], sort='time')

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    df_combined['gt'] = df_combined['temp'].shift(-1, fill_value=21)
    df_combined['day'] = df_combined.index.dayofweek
    df_combined['hour'] = df_combined.index.hour

    week_mask = df_combined['day'] <= 4
    df_combined = df_combined.loc[week_mask]

    hour_mask = (df_combined['hour'] >= 4) & (df_combined['hour'] <= 16)
    df_combined = df_combined.loc[hour_mask]

    mask = (df_combined.index > '2020-10-25')
    df_train = df_combined.loc[mask]

    X_train = df_train[['temp', 'valve_level', 'target_temp']].to_numpy()[1:-1]
    y_train = df_train['gt'].to_numpy()[1:-1]

    X_test = df_train[['temp', 'valve_level', 'target_temp']].to_numpy()[-2:]

    reg.set_params(n_estimators=200, warm_start=True)
    reg.fit(X_train, y_train)

    y_predicted_reg_lin = reg.predict(X_test)

    return y_predicted_reg_lin[1]

