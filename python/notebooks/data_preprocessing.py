import pandas as pd
import numpy as np

def initial_data_preprocessing(df_input:pd.DataFrame,minimum_date = '2017-09-19', maximum_date = '2022-05-01') -> pd.DataFrame:
    
    '''
    Main steps of this initial preprocessing:

    - Drop useless columns.
    - Columns renaming.
    - Filter initial date (default = 2017-09-19).
    - Drop rows with all missing.
    - Build a timestamp and an year columns.
    - Wind speed extrapolation to 120m.

    Parameters
    ----------
    
    df_input: dataframe to preprocess

    minimum_date: minimum date to filter dataframe.

    Returns 
    ----------
    df: preprocessed dataframe.

    '''
    
    print("Dataset initial size: ",df_input.shape)
    df = df_input.copy() #making a copy to preserve raw dataframe.

    if 'Unnamed: 22' in df.columns:
        df.drop(columns=['Unnamed: 22'],inplace=True)
    df.drop(columns=['TEMPERATURA DA CPU DA ESTACAO(°C)','TENSAO DA BATERIA DA ESTACAO(V)'],inplace=True)


    df.rename(columns={
    'Data Medicao': 'DATE_MEASUREMENT',
    'Hora Medicao': 'HOUR_MEASUREMENT',
    'PRECIPITACAO TOTAL, HORARIO(mm)': 'TOTAL_PRECIPITATION_mm',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)': 'ATM_PRESSURE_mB',
    'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)': 'ATM_PRESSURE_SEA_LEVEL_mB',
    'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)': 'MAX_ATM_PRESSURE_PREV_HOUR_mB',
    'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)': 'MIN_ATM_PRESSURE_PREV_HOUR_mB',
    'RADIACAO GLOBAL(Kj/m²)': 'GLOBAL_RADIATION_Kjm2',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)': 'AIR_TEMPERATURE_DRY_BULB_Celsius',
    'TEMPERATURA DO PONTO DE ORVALHO(°C)': 'DEW_POINT_TEMPERATURE_Celsius',
    'TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)': 'MAX_TEMPERATURE_PREV_HOUR_Celsius',
    'TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)': 'MIN_TEMPERATURE_PREV_HOUR_Celsius',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)': 'DEW_POINT_MAX_TEMPERATURE_PREV_HOUR_Celsius',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)': 'DEW_POINT_MIN_TEMPERATURE_PREV_HOUR_Celsius',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)': 'MAX_RELATIVE_HUMIDITY_PREV_HOUR_percentage',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)': 'MIN_RELATIVE_HUMIDITY_PREV_HOUR_percentage',
    'UMIDADE RELATIVA DO AR, HORARIA(%)': 'RELATIVE_HUMIDITY_percentage',
    'VENTO, DIRECAO HORARIA (gr)(° (gr))': 'WIND_DIRECTION_degrees',
    'VENTO, RAJADA MAXIMA(m/s)': 'WIND_MAX_GUNS_ms',
    'VENTO, VELOCIDADE HORARIA(m/s)': 'WIND_SPEED_ms'
    },inplace=True)

    df = df.query("DATE_MEASUREMENT >= @minimum_date and DATE_MEASUREMENT <= @maximum_date")

    rows_with_no_data = df.drop(columns=['DATE_MEASUREMENT','HOUR_MEASUREMENT']).isna().all(axis=1) #not considering the date and hour because they are always present.
    df = df.loc[~(rows_with_no_data),:]
    
    df['YEAR'] = df['DATE_MEASUREMENT'].str.split('-').str[0]
    df['MONTH'] = df['DATE_MEASUREMENT'].str.split('-').str[1]
    df['DAY'] = df['DATE_MEASUREMENT'].str.split('-').str[2]
    df['HOUR'] = (df['HOUR_MEASUREMENT'] / 100).astype(int)
    df['DATETIME'] = pd.to_datetime(df['DATE_MEASUREMENT'] + ' ' + df['HOUR'].astype(str) + ':00:00')
    
    df['WIND_SPEED_120m_ms'] = df['WIND_SPEED_ms'] * (120/5) ** 0.14 #wind speed extrapolation to 120m meters of height and alpha = 0.14

    df.drop(columns=['HOUR_MEASUREMENT'],inplace=True)

    print("Dataset final size: ",df.shape)
    return df.reset_index(drop=True)


def prepare_data_for_feature_generation(df,cols_to_fill,fillna_method = 'forward_fill'):
    '''
    Function to prepare dataset for feature generation.
    
    Steps are:

    - Resample dataset in order to maintain hour time structure, since there are some missing hours.

    - Forward fill of missing values. 
    
    Parameters
    ----------

    df: main dataset containing metheorological data from automatic stations.

    fillna_method: Flag to indicate type of missing value filling. Possible methods are: None, 'rolling' or 'forward_fill'. If it is None, no imputation is used.


    Returns
    ---------
    
    '''
    station_codes = df['CODE'].unique()
    station_names = df['NAME'].unique()
    stations_data = []
    for code,name in zip(station_codes,station_names):

        df_code = df.query("CODE == @code")
        size_before = df_code.shape[0]
        print(f"Processing station {code}-{name}. Current size: {size_before} hour points.")
        df_code = resample_hours_for_wind_speed(df_code)

        df_code['HOUR'] = df_code['DATETIME'].dt.hour
        df_code['DAY'] = df_code['DATETIME'].dt.day
        df_code['MONTH'] = df_code['DATETIME'].dt.month
        df_code['YEAR'] = df_code['DATETIME'].dt.year

        df_code['ALTITUDE'] = df_code['ALTITUDE'].iloc[0]
        df_code['LATITUDE'] = df_code['LATITUDE'].iloc[0]
        df_code['LONGITUDE'] = df_code['LONGITUDE'].iloc[0]
        df_code['CODE'] = df_code['CODE'].iloc[0]
        df_code['NAME'] = df_code['NAME'].iloc[0]

        df_code.sort_values(by='DATETIME',ascending=True,inplace=True)
        size_after = df_code.shape[0]
        print(f"Resampled {size_after-size_before} hour points\n")
        if fillna_method == 'forward_fill':
            df_code = forward_fillna(df_code,cols_to_fill)
        elif fillna_method == 'rolling':
            df_code = rolling_average_fillna(df_code,cols_to_fill)

        stations_data.append(df_code)
    df_stations = pd.concat(stations_data)
    return df_stations

def resample_hours_for_wind_speed(df):
    df  = df.set_index('DATETIME').resample('H').first().reset_index('DATETIME')
    return df

def forward_fillna(df,cols_to_fill):
    df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill',limit=2)
    return df

def rolling_average_fillna(df,cols_to_fill):
    df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].rolling(window=6).mean(min_periods=3),limit=2)
    return df

