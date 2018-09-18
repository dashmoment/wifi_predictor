import pandas as pd
import numpy as np
from datetime import datetime
import copy

def transform_delay2category(delay):

    delay = pd.DataFrame(delay)
    delay_cat = pd.cut(
        delay.stack(),
        [-np.inf, 5, 10, 20, np.inf],
        labels=[0, 1, 2, 3]
    ).reset_index(drop=True)

    return delay_cat


def transfer_to_one_hot_class(delay):

    delay = pd.DataFrame(delay)
    delay_cat = transform_delay2category(delay)
    delay_cat.rename('delay_mean', inplace=True)
    delay_cat = pd.DataFrame(delay_cat)
    delay_cat = pd.get_dummies(delay_cat['delay_mean'])

    return delay_cat


def random_sample(inputs, label_inputs, fraction=0.8):

    total = pd.DataFrame(inputs)
    label_total = pd.DataFrame(label_inputs)

    train = total.sample(frac=fraction)
    sample_idx = train.index
    label_train = label_total.iloc[sample_idx]

    test = total[~total.index.isin(sample_idx)]
    label_test = label_total[~total.index.isin(sample_idx)]

    return train, label_train, test, label_test


def random_sample_conti(inputs, label_inputs, fraction=0.8):

    total = pd.DataFrame(inputs)
    label_total = pd.DataFrame(label_inputs)

    num_sample = int(total.shape[0]*fraction)-1
    cut_start = int(np.random.choice(np.arange(num_sample), 1))
    cut_end = cut_start + int(total.shape[0]*(1-fraction))

    test = total.iloc[cut_start:cut_end, :]
    sample_idx = test.index
    label_test = label_total.iloc[sample_idx]

    train = total[~total.index.isin(sample_idx)]
    label_train = label_total[~total.index.isin(sample_idx)]

    return train, label_train, test, label_test


def sort_by_time(raw_data, raw_label_data):
    
    temp = pd.DataFrame(raw_data)
    data = temp.copy(deep=True)
    data.Time = data.Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))

    data_sorted = data.sort_values(by='Time').reset_index()
    data_sorted_notime = data_sorted.drop(['index', 'Time'], axis=1)
    data_sorted_notime.dropna(axis=0, inplace=True)
    data_sorted_notime.reset_index(drop=True, inplace=True)
    
    label_data = copy.deepcopy(raw_label_data)
    label_data['delay_mean'].Time = label_data['delay_mean'].Time.apply(
        lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
    label_data_sorted = label_data['delay_mean'].sort_values(by='Time').reset_index()
    label_data_sorted.columns = ['index', 'Time', 'delay_mean']
    label_data_sorted.dropna(axis=0, inplace=True)
    label_data_sorted.reset_index(drop=True, inplace=True)

    return data_sorted_notime, label_data_sorted


def weighted_random_sample(inputs, label_inputs, fraction=0.8):
    pass
