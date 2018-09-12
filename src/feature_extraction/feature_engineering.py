import pandas as pd
import numpy as np


def binding(dataframe=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    # extract subvalue of AP and STA
    ap_sub = [col for col in dataframe.columns if ('Subval' in col and 'AP' in col)]
    sta_sub = [col for col in dataframe.columns if ('Subval' in col and 'STA' in col)]

    # df_ap = dataframe[ap_sub]
    # df_sta = dataframe[sta_sub]

    # generating feature
    (dataframe.pipe(ComputeVar, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeVar, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMean, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMean, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeShannon, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeShannon, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMax, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMax, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMin, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMin, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMaxMinusMin, client='AP', ap_sub=ap_sub, sta_sub=sta_sub)
              .pipe(ComputeMaxMinusMin, client='STA', ap_sub=ap_sub, sta_sub=sta_sub)
     )

    dataframe.drop(ap_sub + sta_sub, axis=1, inplace=True)

    return dataframe


def ComputeShannon(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = np.apply_along_axis(lambda x: sum(-np.abs(x)*np.log(np.abs(x))), axis=1, arr=df_ap)
        dataframe['AP-Sub-Shannon'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = np.apply_along_axis(lambda x: sum(-np.abs(x)*np.log(np.abs(x))), axis=1, arr=df_sta)
        dataframe['STA-Sub-Shannon'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe


def ComputeVar(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = df_ap.var(axis=1)
        dataframe['AP-Sub-Var'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = df_sta.var(axis=1)
        dataframe['STA-Sub-Var'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe


def ComputeMean(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = df_ap.mean(axis=1)
        dataframe['AP-Sub-Mean'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = df_sta.mean(axis=1)
        dataframe['STA-Sub-Mean'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe


def ComputeMax(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = df_ap.max(axis=1)
        dataframe['AP-Sub-Max'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = df_sta.max(axis=1)
        dataframe['STA-Sub-Max'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe


def ComputeMin(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = df_ap.min(axis=1)
        dataframe['AP-Sub-Min'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = df_sta.min(axis=1)
        dataframe['STA-Sub-Min'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe


def ComputeMaxMinusMin(dataframe, client=None, ap_sub=None, sta_sub=None):

    assert isinstance(dataframe, pd.DataFrame) == True

    if client == 'AP':
        df_ap = dataframe[ap_sub].values
        feature_ap = np.apply_along_axis(lambda x: np.max(x)-np.min(x), axis=1, arr=df_ap)
        dataframe['AP-Sub-Max-min'] = pd.Series(feature_ap, index=dataframe.index)

    if client == 'STA':
        df_sta = dataframe[sta_sub].values
        feature_sta = np.apply_along_axis(lambda x: np.max(x)-np.min(x), axis=1, arr=df_sta)
        dataframe['STA-Sub-Max-min'] = pd.Series(feature_sta, index=dataframe.index)

    return dataframe