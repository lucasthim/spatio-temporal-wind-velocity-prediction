import pandas as pd
import numpy as np

def make_experiment_dataset(df:pd.DataFrame,hourly_features:list,lags:list,rolling_windows:list, hour_feature:str, targets:list, target_shift = 1)-> pd.DataFrame:
    '''
    
    Builds the features and target to execute the wind prediction experiment.

    Main features types are:

    - Lagged versions of features.
    - Central Tendency features: mean and median.
    - Minimum and Maximum of features.
    - Dispersion features: standard deviation.
    - Hour of the target.

    Targets are shifted hours ahead, controlled by the variable target_shift.

    Parameters
    ----------

    df: dataset containing features
    
    hourly_features: features containing data that is registered every hour.

    lags: Lags to generate lagged features. If lags should not be calculated, pass None to the variable.

    rolling_windows: Rolling windows to generate central tendency, dispersion and min max features. If no rolling features should be calculated, pass None to the variable.

    hour_feature: indication of the hour feature column to generate the hour of the target feature. If this feature should be calculated, pass None to the variable.
    
    targets: columns indicating the targets in the dataset.

    target_shift: List of number indicating how many hours ahead the targets are. 

    cols_to_drop: Columns to drop after all the features and target are generated.

    Returns
    ---------
    df_feature_data: dataframe containing all the generated features

    '''
    station_codes = df['CODE'].unique()
    final_data = []
    for code in station_codes:
        df_code = df.query("CODE == @code").sort_values(by='DATETIME',ascending=True)
        df_features = make_features(df_code,hourly_features,lags,rolling_windows,hour_feature,target_shift)
        df_target = make_target(df_code[targets].copy(),targets,target_shift)
        df_code_dataset = pd.concat([df_features,df_target],axis=1)
        final_data.append(df_code_dataset)

    df_final_dataset = pd.concat(final_data)
    df_final_dataset = drop_duplicate_columns(df_final_dataset)
    return df_final_dataset

def make_features(df:pd.DataFrame,hourly_features:list,lags:list,rolling_windows:list, hour_feature = 'HOUR', target_shift = 1) -> pd.DataFrame:
    '''
    Makes the time series features to feed to the model.

    Main features types are:

    - Lagged versions of features.
    - Central Tendency features: mean and median.
    - Minimum and Maximum of features.
    - Dispersion features: standard deviation.
    - Hour of the target.

    Parameters
    ----------

    df: dataset containing features
    
    hourly_features: features containing data that is registered every hour.

    lags: Lags to generate lagged features. If lags should not be calculated, pass None to the variable.

    rolling_windows: Rolling windows to generate central tendency, dispersion and min max features. If no rolling features should be calculated, pass None to the variable.

    target_shift: List of number indicating how many hours ahead the targets are. 

    hour_feature: indication of the hour feature column to generate the hour of the target feature. If this feature should be calculated, pass None to the variable.

    Returns
    ---------
    df: dataframe containing all the generated features

    '''

    if lags != [] and lags is not None:
        df = make_lag_features(df,lags=lags,features=hourly_features)
    
    if rolling_windows != [] and rolling_windows is not None:
        df = make_min_max_features(df,features=hourly_features,windows=rolling_windows)
        df = make_central_tendency_features(df,features=hourly_features,windows=rolling_windows)
        df = make_dispersion_features(df,features=hourly_features,windows=rolling_windows)
    
    if hour_feature != '' and hour_feature is not None:
        df = make_target_hour_feature(df,features=[hour_feature],target_shift=target_shift)
    return df

def make_lag_features(df,lags,features):
    '''Shifts backwards(to the past) the chosen features in order to create lagged versions of them.'''
    for lag in lags:
        df[[t + f'_lag_{lag}h' for t in features]] = df[features].shift(lag)
    return df
    
def make_min_max_features(df,features,windows):
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_min_{window}' for feature in features]] = df[features].rolling(window=window).min(min_periods=minimum_period)
        df[[feature+f'_max_{window}' for feature in features]] = df[features].rolling(window=window).max(min_periods=minimum_period)
    return df

def make_central_tendency_features(df,features,windows):
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_mean_{window}' for feature in features]] = df[features].rolling(window=window).mean(min_periods=minimum_period)
        df[[feature+f'_median_{window}' for feature in features]] = df[features].rolling(window=window).median(min_periods=minimum_period)
    return df

def make_dispersion_features(df,features,windows):
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_std_{window}' for feature in features]] = df[features].rolling(window=window).std(min_periods=minimum_period)
    return df

def make_target_hour_feature(df,features,target_shift=1):
    '''Apply the same target temporal shift to the hour feature.'''
    return make_target(df,features,target_shift)
    
def make_target(df,targets,target_shift = 1):
    '''
    Shifts temporal data to make the target variables. 

    Parameters
    ----------

    df: dataset containing raw targets.

    targets: columns indicating the targets in the dataset.

    target_shift: List or number indicating how many hours ahead the targets should be shifted. 

    Returns
    ---------
    df_target: dataframe containing all the generated targets.
    '''

    if type(target_shift) == int:
        df[[t + f'_target_{target_shift}h' for t in targets]] = df[targets].shift(-target_shift)

    elif type(target_shift) == list:
        for target in target_shift:
            df[[t + f'_target_{target}h' for t in targets]] = df[targets].shift(-target)
    return df

def drop_duplicate_columns(df):
    columns = list(set(df.columns.tolist()))
    return df[columns]