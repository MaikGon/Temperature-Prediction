import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sktime
# import fbprophet
import pmdarima
import statsmodels.tsa
import pyts
import json
import random
import pickle

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
def read_temp(where):
    with open('dataset/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']

    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == where]
    # print("Mid serial number: ", sn_temp_mid)
    return sn_temp_mid[0]


def read_data(name):
    df = pd.read_csv('dataset/' + name)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.drop(columns=['unit'], inplace=True)
    df.set_index('time', inplace=True)

    return df


def wall():
    sn_temp_wall = read_temp('temperature_wall')

    # Temperature
    df_temp_1 = read_data('office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_temp_2 = read_data('office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp = pd.concat([df_temp_1, df_temp_2])
    df_temp.rename(columns={'value': 'temp'}, inplace=True)

    # Serial number for middle
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_wall]

    # Target
    df_target_1 = read_data('office_1_targetTemperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_target_2 = read_data('office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target = pd.concat([df_target_1, df_target_2])
    df_target.rename(columns={'value': 'target_temp'}, inplace=True)

    # Valve
    df_valve_1 = read_data('office_1_valveLevel_supply_points_data_2020-03-05_2020-03-19.csv')
    df_valve_2 = read_data('office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve = pd.concat([df_valve_1, df_valve_2])
    df_valve.rename(columns={'value': 'valve_level'}, inplace=True)

    df_combined = pd.concat([df_temp, df_target, df_valve], sort='time')
    # Now let's resample
    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    df_combined['gt'] = df_combined['temp'].shift(-1, fill_value=21)
    df_combined['day'] = df_combined.index.dayofweek
    df_combined['hour'] = df_combined.index.hour

    week_mask = df_combined['day'] <= 4
    df_combined = df_combined.loc[week_mask]

    hour_mask = (df_combined['hour'] >= 4) & (df_combined['hour'] <= 16)
    df_combined = df_combined.loc[hour_mask]

    mask = (df_combined.index < '2020-10-26')
    df_train = df_combined.loc[mask]

    X_train = df_train[['temp', 'valve_level', 'target_temp']].to_numpy()[1:-1]
    y_train = df_train['gt'].to_numpy()[1:-1]

    reg_rf = GradientBoostingRegressor(n_estimators=175)
    reg_rf.fit(X_train, y_train)

    reg_random = RandomForestRegressor()
    reg_random.fit(X_train, y_train)

    # reg_lin = SVR(kernel='poly', C=100)
    reg_lin = LinearRegression()
    reg_lin.fit(X_train, y_train)

    reg_vot = VotingRegressor([('gr', reg_rf), ('lin', reg_lin), ('rf', reg_random)])
    reg_vot.fit(X_train, y_train)

    # Wycinanie jednego dnia do testów
    mask_test = (df_combined.index >= '2020-10-26') & (df_combined.index < '2020-10-27')
    df_test = df_combined.loc[mask_test]

    X_test = df_test[['temp', 'valve_level', 'target_temp']].to_numpy()
    y_test = df_test['gt'].to_numpy()

    y_predicted_reg_rf = reg_rf.predict(X_test)
    df_test['temp_predicted_gradient'] = y_predicted_reg_rf.tolist()

    y_predicted_reg_forest = reg_random.predict(X_test)
    #df_test['temp_predicted_forest'] = y_predicted_reg_forest.tolist()

    y_predicted_reg_lin = reg_lin.predict(X_test)
    #df_test['temp_predicted_lin'] = y_predicted_reg_lin.tolist()

    y_predicted_reg_vot = reg_vot.predict(X_test)
    df_test['temp_predicted_voting'] = y_predicted_reg_vot.tolist()

    print(f'mae gradient: {metrics.mean_absolute_error(y_test, y_predicted_reg_rf)}')
    print(f'mae forest: {metrics.mean_absolute_error(y_test, y_predicted_reg_forest)}')
    print(f'mae voting: {metrics.mean_absolute_error(y_test, y_predicted_reg_vot)}')
    print(f'mae linear: {metrics.mean_absolute_error(y_test, y_predicted_reg_lin)}')

    pickle.dump(reg_rf, open('./clf.p', 'wb'))

    df_test.drop(columns=['valve_level', 'target_temp', 'day', 'hour'], inplace=True)
    df_test.plot()

    # df_combined.plot()
    # plt2 = plt.twinx()
    # plt2.plot(df_valve.index, df_valve.value, color='g')
    plt.show()


def middle():
    pass


def window():
    pass


if __name__ == "__main__":
    temp = 'office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv'
    target = 'office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv'
    valve = 'office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv'

    wall()
