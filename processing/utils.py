import pandas as pd
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple

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
random.seed(42)


def perform_processing(temperature: pd.DataFrame, target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,serial_number_for_prediction: str) -> Tuple[float, float]:

    # NOTE(MF): sample how this can be done
    # data = preprocess_data(temperature, target_temperature, valve_level)
    # model_temp, model_valve = load_models()
    # or load model once at the beginning and pass it to this function
    # predicted_temp = model_temp.predict(data)
    # predicted_valve_level = model_valve.predict(data)
    # return predicted_temp, predicted_valve_level

    with Path('clf.p').open('rb') as reg_file:
        reg = pickle.load(reg_file)

    with Path('valve.p').open('rb') as valve_file:
        valve_reg = pickle.load(valve_file)

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
    df_combined['gt_valve'] = df_combined['valve_level'].shift(-1, fill_value=21)
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

    # temperature refit
    reg.set_params(n_estimators=160, warm_start=True)
    reg.fit(X_train, y_train)
    y_predicted_reg = reg.predict(X_test)

    X_train_valve = df_train[['temp', 'valve_level', 'target_temp']].to_numpy()[1:-1]
    y_train_valve = df_train['gt_valve'].to_numpy()[1:-1]

    X_test_valve = df_train[['temp', 'valve_level', 'target_temp']].to_numpy()[-2:]

    # valve refit
    valve_reg.set_params(n_estimators=90, warm_start=True)
    valve_reg.fit(X_train_valve, y_train_valve)
    y_predicted_valve = valve_reg.predict(X_test_valve)

    # return temp and valve
    return y_predicted_reg[1], y_predicted_valve[1]

