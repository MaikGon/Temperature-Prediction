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
import fbprophet
import pmdarima
import statsmodels.tsa
import pyts
import json
import random

random.seed(42)


def load_data():
    with open('dataset/additional_info.json') as f:
        additional_data = json.load(f)

    print(additional_data)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_nid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle']
    print(sn_temp_nid)


if __name__ == "__main__":
    load_data()
