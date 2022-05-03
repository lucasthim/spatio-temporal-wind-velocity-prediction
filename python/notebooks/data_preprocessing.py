import pandas as pd
import numpy as np

def initial_data_preprocessing(df_input:pd.DataFrame,minimum_date = '2017-09-19',imputing_window=1) -> pd.DataFrame:
    '''
    Main steps of this initial preprocessing:

    - Drop useless columns.
    - Column name normalization.
    - Filter initial date (default = 2017-09-19).
    - Drop rows with all missing.
    - Build a timestamp and an year columns.
    - Wind speed extrapolation to 120m.
    X. Basic data imputation (roll forward or interpolation for 1h).

    Parameters
    ----------
    
    df_input: dataframe to preprocess

    minimum_date: minimum date to filter dataframe.

    imputing_window: forward window to impute data. The value will propagate forward in case of missing value. (This does not apply on the target variables)


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

    df = df.query("DATE_MEASUREMENT >= @minimum_date")

    rows_with_no_data = df.drop(columns=['DATE_MEASUREMENT','HOUR_MEASUREMENT']).isna().all(axis=1) #not considering the date and hour because they are always present.
    df = df.loc[~(rows_with_no_data),:]
    
    df['YEAR'] = df['DATE_MEASUREMENT'].str.split('-').str[0]
    df['HOUR_MEASUREMENT'] = (df['HOUR_MEASUREMENT'] / 100).astype(int)
    df['DATETIME'] = pd.to_datetime(df['DATE_MEASUREMENT'] + ' ' + df['HOUR_MEASUREMENT'].astype(str) + ':00:00')


    # TODO: Data imputation

    df['WIND_SPEED_120m_ms'] = df['WIND_SPEED_ms'] * (120/5) ** 0.14 #wind speed extrapolation to 120m meters of height and alpha = 0.14

    print("Dataset final size: ",df.shape)
    return df.reset_index(drop=True)