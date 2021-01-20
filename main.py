import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import json
import random
import pickle

random.seed(42)


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
    sn_temp_middle = read_temp('temperature_middle')
    sn_temp_window = read_temp('temperature_window')

    # Temperature
    df_temp_1 = read_data('office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_temp_2 = read_data('office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp = pd.concat([df_temp_1, df_temp_2])
    df_temp.rename(columns={'value': 'temp'}, inplace=True)

    # Serial number
    df_temp_1 = df_temp[df_temp['serialNumber'] == sn_temp_wall]
    df_temp_1.rename(columns={'temp': 'temp_wall'}, inplace=True)
    df_temp_2 = df_temp[df_temp['serialNumber'] == sn_temp_middle]
    df_temp_2.rename(columns={'temp': 'temp_middle'}, inplace=True)
    df_temp_3 = df_temp[df_temp['serialNumber'] == sn_temp_window]
    df_temp_3.rename(columns={'temp': 'temp_window'}, inplace=True)

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

    df_combined = pd.concat([df_temp_1, df_temp_2, df_temp_3, df_target, df_valve], sort='time')
    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    df_combined['gt'] = df_combined['temp_middle'].shift(-1, fill_value=21)
    df_combined['gt_valve'] = df_combined['valve_level'].shift(-1, fill_value=40)
    df_combined['day'] = df_combined.index.dayofweek
    df_combined['hour'] = df_combined.index.hour

    week_mask = df_combined['day'] <= 4
    df_combined = df_combined.loc[week_mask]

    hour_mask = (df_combined['hour'] >= 4) & (df_combined['hour'] <= 16)
    df_combined = df_combined.loc[hour_mask]

    mask = (df_combined.index < '2020-10-23')
    df_train = df_combined.loc[mask]

    X_train = df_combined[['temp_window', 'temp_middle', 'valve_level', 'target_temp']].to_numpy()[1:-1]
    y_train = df_combined['gt'].to_numpy()[1:-1]

    reg_rf = GradientBoostingRegressor(n_estimators=80)
    reg_rf.fit(X_train, y_train)

    X_train_valve = df_combined[['temp_wall', 'temp_window', 'temp_middle', 'valve_level', 'target_temp']].to_numpy()[1:-1]
    y_train_valve = df_combined['gt_valve'].to_numpy()[1:-1]

    base_es = GradientBoostingRegressor()
    param_grid = [
        {'n_estimators': [50, 80, 100, 120, 130, 140, 170, 200, 220, 250, 300]}
    ]

    # grid_s = GridSearchCV(base_es, param_grid).fit(X_train_valve, y_train_valve)
    # print(grid_s.best_params_)
    reg_rf_valve = GradientBoostingRegressor(n_estimators=40)
    reg_rf_valve.fit(X_train_valve, y_train_valve)

    # Wycinanie jednego dnia do testów
    mask_test = (df_combined.index >= '2020-10-23') & (df_combined.index < '2020-10-24')
    df_test = df_combined.loc[mask_test]

    X_test = df_test[['temp_window', 'temp_middle', 'valve_level', 'target_temp']].to_numpy()
    y_test = df_test['gt'].to_numpy()

    y_predicted_reg_rf = reg_rf.predict(X_test)
    df_test['temp_predicted_gradient'] = y_predicted_reg_rf.tolist()

    print(f'mae temperature: {metrics.mean_absolute_error(y_test, y_predicted_reg_rf)}')

    X_test_valve = df_test[['temp_wall','temp_window', 'temp_middle', 'valve_level', 'target_temp']].to_numpy()
    y_test_valve = df_test['gt_valve'].to_numpy()

    y_predicted_reg_rf_valve = reg_rf_valve.predict(X_test_valve)
    df_test['temp_predicted_gradient_valve'] = y_predicted_reg_rf_valve.tolist()

    print(f'mae valve: {metrics.mean_absolute_error(y_test_valve, y_predicted_reg_rf_valve)}')

    pickle.dump(reg_rf, open('./model/temp_model1.p', 'wb'))
    pickle.dump(reg_rf_valve, open('./model/valve_model1.p', 'wb'))

    df1 = df_test.drop(columns=['temp_wall', 'temp_middle', 'temp_window',  'valve_level', 'target_temp', 'gt_valve', 'temp_predicted_gradient_valve', 'day', 'hour'])
    df1.plot()

    df2 = df_test.drop(columns=['temp_wall', 'temp_middle', 'temp_window', 'gt', 'valve_level', 'temp_predicted_gradient', 'target_temp', 'day', 'hour'])
    df2.plot()

    plt.show()


if __name__ == "__main__":
    wall()
