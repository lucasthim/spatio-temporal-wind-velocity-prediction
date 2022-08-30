import pandas as pd
import numpy as np

def make_wind_prediction_dataset(df:pd.DataFrame,granularity:str, main_features:list,lags:list,rolling_windows:list, targets:list, target_shift = 1)-> pd.DataFrame:
    
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
    
    main_features: features containing data that is registered every hour.

    lags: Lags to generate lagged features. If lags should not be calculated, pass None to the variable.

    rolling_windows: Rolling windows to generate central tendency, dispersion and min max features. If no rolling features should be calculated, pass None to the variable.

    granularity: string indicating the granularity of data to produce the features. Options are 'HOUR','DAY'
    
    targets: columns indicating the targets in the dataset.

    target_shift: List of number indicating how many hours ahead the targets are. 

    cols_to_drop: Columns to drop after all the features and target are generated.

    Returns
    ---------
    df_feature_data: dataframe containing all the generated features

    '''

    if 'DATETIME' not in df.columns:
        raise ValueError("DATETIME column not found in provided DataFrame.")

    if granularity.upper() not in ['DAY','HOUR']:
        raise ValueError("Value provided for granularity is not valid. Options are 'day' or 'hour'.")
        
    station_codes = df['CODE'].unique()
    final_data = []
    for code in station_codes:
        df_code = df.query("CODE == @code").sort_values(by='DATETIME',ascending=True)
        if granularity.upper() == 'DAY':
            df_code = change_granularity_of_dataset_to_day(df_code)

        df_features = make_features(df_code,granularity,main_features,lags,rolling_windows,target_shift)
        df_target = make_target(df_code[targets].copy(),granularity,targets,target_shift)
        df_target.drop(columns=targets,inplace=True) #dropping former name of targets.
        df_code_dataset = pd.concat([df_features,df_target],axis=1)
        final_data.append(df_code_dataset)

    df_final_dataset = pd.concat(final_data)
    return df_final_dataset

def change_granularity_of_dataset_to_day(df):

    code = df['CODE'].iloc[0]
    name = df['CODE'].iloc[0]
    
    df['DATE'] = pd.to_datetime(df['DATETIME'].dt.date)
    df.drop(columns=['DATETIME'],inplace=True)
    df_count = df.set_index('DATE').resample('D').agg('count')
    df_mean = df.set_index('DATE').resample('D').agg('mean')

    df_mean[df_count < 12] = np.nan # in case the day has less than 12 points, replace for null.
    df_mean['CODE'] = code
    df_mean['NAME'] = name

    return df_mean.reset_index().drop(columns=['HOUR'])

def make_features(df:pd.DataFrame, granularity:str, main_features:list, lags:list, rolling_windows:list, target_shift = 1) -> pd.DataFrame:
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
    
    main_features: features containing data that is registered every hour.

    lags: Lags to generate lagged features. If lags should not be calculated, pass None to the variable.

    rolling_windows: Rolling windows to generate central tendency, dispersion and min max features. If no rolling features should be calculated, pass None to the variable.

    target_shift: List of number indicating how many hours ahead the targets are. 

    granularity: indication of the hour feature column to generate the hour of the target feature. If this feature should be calculated, pass None to the variable.

    Returns
    ---------
    df: dataframe containing all the generated features
    '''

    if lags != [] and lags is not None:
        df = make_lag_features(df,granularity,lags=lags,features=main_features)
    
    if rolling_windows != [] and rolling_windows is not None:
        df = make_min_max_features(df,granularity,features=main_features,windows=rolling_windows)
        df = make_central_tendency_features(df,granularity,features=main_features,windows=rolling_windows)
        df = make_dispersion_features(df,granularity,features=main_features,windows=rolling_windows)
    
    # TODO: Mudar aqui
    if granularity != '' and granularity is not None:
        df = make_target_time_feature(df,granularity,target_shift=target_shift)

    return df

def make_lag_features(df,granularity,lags,features):
    '''Shifts backwards(to the past) the chosen features in order to create lagged versions of them.'''

    time_suffix = get_time_suffix(granularity)
    for lag in lags:
        df[[t + f'_lag_{lag}{time_suffix}' for t in features]] = df[features].shift(lag)
    return df
    
def make_min_max_features(df,granularity,features,windows):
    time_suffix = get_time_suffix(granularity)
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_min_{window}{time_suffix}' for feature in features]] = df[features].rolling(window=window).min(min_periods=minimum_period)
        df[[feature+f'_max_{window}{time_suffix}' for feature in features]] = df[features].rolling(window=window).max(min_periods=minimum_period)
    return df

def make_central_tendency_features(df,granularity,features,windows):
    time_suffix = get_time_suffix(granularity)
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_mean_{window}{time_suffix}' for feature in features]] = df[features].rolling(window=window).mean(min_periods=minimum_period)
        df[[feature+f'_median_{window}{time_suffix}' for feature in features]] = df[features].rolling(window=window).median(min_periods=minimum_period)
    return df

def make_dispersion_features(df,granularity,features,windows):
    time_suffix = get_time_suffix(granularity)
    for window in windows:
        minimum_period = int(np.round(window/2))
        df[[feature+f'_std_{window}{time_suffix}' for feature in features]] = df[features].rolling(window=window).std(min_periods=minimum_period)
    return df

def make_target_time_feature(df,granularity,target_shift=1):
    '''Apply the same target temporal shift to the time (hour or day) feature.'''
    return make_target(df,granularity=granularity,targets=[granularity],target_shift=target_shift)


def make_target(df,granularity, targets, target_shift = 1):
    '''
    Shifts temporal data to make the target variables. 

    Parameters
    ----------

    df: dataset containing raw targets.

    granularity: string indicating the granularity of data. Options are 'HOUR','DAY'.

    targets: columns indicating the targets in the dataset.

    target_shift: List or number indicating how many hours ahead the targets should be shifted. 

    Returns
    ---------
    df_target: dataframe containing all the generated targets.
    '''

    time_suffix = get_time_suffix(granularity)
    
    if type(target_shift) == int:
        df[[t + f'_target_{target_shift}{time_suffix}' for t in targets]] = df[targets].shift(-target_shift)

    elif type(target_shift) == list:
        for target in target_shift:
            df[[t + f'_target_{target}{time_suffix}' for t in targets]] = df[targets].shift(-target)
    return df

def get_time_suffix(granularity):

    time_suffix = 'h'
    if granularity.upper() == 'DAY':
        time_suffix = 'd'
    return time_suffix

def drop_null_features_and_instances(df,feature_null_percentage=0.2,instance_null_percentage=0.8):
    '''Drop features and instances that are above a null percentage.'''

    dataset_initial_size = df.shape[0]
    
    print("Initial dataset size:",df.shape)
    nulls = df.isnull().sum() / dataset_initial_size
    nulls_percent = nulls[nulls > feature_null_percentage]
    print(f"There were {nulls_percent.shape[0]} features with more than {feature_null_percentage*100}% of null values:")
    
    cols_nulls_percent = nulls_percent.index.tolist()
    non_null_tresh = np.round(df.drop(columns=cols_nulls_percent).shape[1])
    df_selected = df.drop(columns=cols_nulls_percent)
    total_instances = df_selected.shape[0]
    
    df_selected = df_selected.dropna(axis=0,thresh=non_null_tresh* instance_null_percentage)
    processed_instances = df_selected.shape[0]
    
    print(f"There were a total of {total_instances-processed_instances} with less than {instance_null_percentage*100}% of avaiable data (features).")
    return df_selected