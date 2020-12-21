import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
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


def read_temp_mid_sn():
    with open('dataset/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle']
    print("Mid serial number: ", sn_temp_mid)
    return sn_temp_mid


def project_check_data():
    sn_temp_mid = read_temp_mid_sn()
    df_temp = pd.read_csv('dataset/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    print(df_temp.info())
    print(df_temp.describe())
    print(df_temp.head(5))


if __name__ == "__main__":
    # example_tasks()
    project_check_data()
