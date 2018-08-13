import pandas as pd
import numpy as np
import pickle


def binding(dataframe=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    # extract subvalue of AP and STA
    ap_sub = [col for col in dataframe.columns if ('Subval' in col and 'AP' in col)]
    sta_sub = [col for col in dataframe.columns if ('Subval' in col and 'STA' in col)]

    # df_ap = dataframe[ap_sub]
    # df_sta = dataframe[sta_sub]

    # generating feature
    (dataframe.pipe(ComputeShannon, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeShannon, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
     )

    return dataframe


def ComputeShannon(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub]
        shannon_ap = df_ap.apply(lambda x: sum(-np.abs(x)*np.log(np.abs(x))), axis=1)
        dataframe['AP-Shannon'] = pd.Series(shannon_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub]
        shannon_sta = df_sta.apply(lambda x: sum(-np.abs(x)*np.log(np.abs(x))), axis=1)
        dataframe['STA-Shannon'] = pd.Series(shannon_sta, index=dataframe.index)

    return dataframe


with open('../../data/raw_data_wo_time_sub.pkl', 'rb') as input:

    train_raw = pickle.load(input)
    label_train_raw = pickle.load(input)
    test_raw = pickle.load(input)
    label_test_raw = pickle.load(input)

output = binding(test_raw)
print(output.head(3))
