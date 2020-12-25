import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm, tree
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sktime
# import fbprophet
import pmdarima
import statsmodels.tsa
import pyts
import json
import random

random.seed(42)


# Training
def example_tasks():
    dates = pd.date_range('20191204', periods=30, freq='5s')
    df = pd.DataFrame(
        {'power': np.random.randint(low=0, high=50, size=len(dates))},
        index=dates
    )

    df_resampled_mean = df.resample('6s').mean()
    df_resampled_nearest = df.resample('6s').nearest()
    df_resampled_ffill = df.resample('6s').ffill()
    df_resampled_bfill = df.resample('6s').bfill()

    df_plt, = plt.plot(df)
    mean_plt, = plt.plot(df_resampled_mean)
    nearest_plt, = plt.plot(df_resampled_nearest)
    ffill_plt, = plt.plot(df_resampled_ffill)
    bfill_plt, = plt.plot(df_resampled_bfill)
    plt.legend(
        [df_plt, mean_plt, nearest_plt, ffill_plt, bfill_plt],
        ['df', 'mean', 'nearest', 'ffill', 'bfill'],
        loc='upper right'
    )
    plt.show()


# Project part
def read_temp_mid_sn():
    with open('dataset/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']

    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle']
    print("Mid serial number: ", sn_temp_mid)
    return sn_temp_mid[0]


def read_data(name):
    df = pd.read_csv('dataset/' + name)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.drop(columns=['unit'], inplace=True)
    df.set_index('time', inplace=True)

    return df


def project_check_data():
    sn_temp_mid = read_temp_mid_sn()
    print("Mid serial number: ", sn_temp_mid)

    # Temperature
    df_temp = read_data('office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')

    # Serial number for middle
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]

    # Target
    df_target = read_data('office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target.rename(columns={'value': 'target_temp'}, inplace=True)
    # Valve
    df_valve = read_data('office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve.rename(columns={'value': 'valve_level'}, inplace=True)

    # Show some info
    # print(df_temp.info())
    # print(df_temp.describe())
    # print(df_temp.head(5))

    df_combined = pd.concat([df_temp, df_target, df_valve], sort='time')
    # Now let's resample
    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    df_combined['temp_last'] = df_combined['value'].shift(periods=1, fill_value=20)
    df_combined['temp_groundTruth'] = df_combined['value'].shift(periods=-1, fill_value=20.34)

    mask = (df_combined.index <= '2020-10-27') | (df_combined.index > '2020-10-28')
    df_train = df_combined.loc[mask]
    #plt.figure()
    #df_train.plot()

    X_train = df_train[['value', 'valve_level']].to_numpy()[1:-1]
    y_train = df_train['temp_groundTruth'].to_numpy()[1:-1]
    reg_rf = RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)

    # Wycinanie jednego dnia
    mask = (df_combined.index > '2020-10-27') & (df_combined.index <= '2020-10-28')
    df_test = df_combined.loc[mask]

    X_test = df_test[['value', 'valve_level']].to_numpy()
    y_predicted = reg_rf.predict(X_test)
    df_test['temp_predicted'] = y_predicted.tolist()

    y_test = df_test['temp_groundTruth'].to_numpy()[1:-1]
    y_last = df_test['temp_last'].to_numpy()[1:-1]

    print(f'mae base: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae forest: {metrics.mean_absolute_error(y_test, y_predicted[1:-1])}')
    print(f'mse base: {metrics.mean_squared_error(y_test, y_last)}')
    print(f'mse forest: {metrics.mean_squared_error(y_test, y_predicted[1:-1])}')

    df_test.drop(columns=['value', 'valve_level', 'target_temp'], inplace=True)
    df_test.plot()
    print(df_combined.head(5))
    print(df_combined.tail(5))

    df_combined.plot()

    #plt.plot(df_temp.index, df_temp.value)
    #plt.plot(df_target.index, df_target.target_temp)

    plt2 = plt.twinx()
    # plt2.plot(df_valve.index, df_valve.value, color='g')
    plt.show()


if __name__ == "__main__":
    temp = 'office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv'
    target = 'office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv'
    valve = 'office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv'

    project_check_data()
