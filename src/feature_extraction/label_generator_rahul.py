import pandas as pd
import numpy as np


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



def weighted_random_sample(inputs, label_inputs, fraction=0.8):
    pass