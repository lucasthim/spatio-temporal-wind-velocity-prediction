import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def setup_datasets_for_experiment(df,test_year=2020,validation_year=2019,target_format='1h',target_name='WIND_SPEED_ms',targets_to_drop=[]):

    df_train_initial = df.query("DATETIME.dt.year != @test_year")
    df_validation = df_train_initial.query("DATETIME.dt.year == @validation_year")
    df_train = df_train_initial.query("DATETIME.dt.year != @validation_year")
    df_test = df.query("DATETIME.dt.year == @test_year")

    features_to_drop = ['NAME', 'CODE', 'DATETIME', 'YEAR']

    if target_format == '1h':
        features_to_drop = features_to_drop + ['HOUR_target_3h', 'HOUR_target_6h']
    elif target_format == '3h':
        features_to_drop = features_to_drop + ['HOUR_target_1h', 'HOUR_target_6h']
    elif target_format == '6h':
        features_to_drop = features_to_drop + ['HOUR_target_3h', 'HOUR_target_1h']
    
    if targets_to_drop == []:
        targets_to_drop = ['WIND_SPEED_ms_target_3h','WIND_SPEED_ms_target_6h', 
                            'WIND_DIRECTION_degrees_target_1h', 'WIND_DIRECTION_degrees_target_3h', 
                            'WIND_DIRECTION_degrees_target_6h', 'WIND_MAX_GUNS_ms_target_1h',
                            'WIND_MAX_GUNS_ms_target_3h', 'WIND_MAX_GUNS_ms_target_6h']

    target = [target_name+ '_target_' + target_format]

    X_train,y_train = feature_target_split(df_train.dropna(subset=target).fillna(0),features_to_drop,targets_to_drop,target)
    X_test,y_test = feature_target_split(df_test.dropna(subset=target).fillna(0),features_to_drop,targets_to_drop,target)

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    if validation_year is not None:
        X_validation,y_validation = feature_target_split(df_validation.dropna(subset=target).fillna(0),features_to_drop,targets_to_drop,target)
        X_validation_norm = scaler.transform(X_validation)
        return X_train_norm,y_train,X_validation_norm,y_validation,X_test_norm,y_test

    return X_train_norm, y_train, X_test_norm, y_test

def train_test_split_by_year(df,test_year=2020) -> tuple[pd.DataFrame]:
    df_train = df.query("DATETIME.dt.year != @test_year")
    df_test = df.query("DATETIME.dt.year == @test_year")
    return df_train,df_test

def feature_target_split(df,features_to_drop,targets_to_drop,target) -> tuple[pd.DataFrame]:
    X = df.drop(columns=(features_to_drop+targets_to_drop+target))
    y = df[target]
    return X,y

def evaluate_predictions(y_pred,y_true,max_prediction,min_prediction=0,set='Train'):

    y_pred = treat_output(y_pred,max=max_prediction,min=min_prediction)
    _ = calculate_erros(y_pred = y_pred,y_true = y_true)
    compare_distributions(y_pred,y_true,set)

def treat_output(y_pred,max,min):
    y_pred[y_pred > max] = 1.5 * max
    y_pred[y_pred<min] = 0.5 * min
    return y_pred

def calculate_erros(y_pred,y_true):

    if type(y_true) != np.ndarray:
        y_true = y_true.values
    mae = mean_absolute_error(y_pred, y_true)
    rmse = mean_squared_error(y_pred, y_true,squared=False)

    print(f"Target distribution [mean(std)]: {y_true.mean(): .2f} ({y_true.std():.2f})")
    print(f"Prediction distribution [mean(std)]: {y_pred.mean(): .2f} ({y_pred.std():.2f})")
    print(f"MAE {mae:.2f}")
    print(f"RMSE {rmse:.2f}")
    print("")
    return mae,rmse

def compare_distributions(y_pred,y_true,set):
    df = pd.DataFrame(columns=['Target','Prediction'])
    df['Target'] = y_true
    df['Prediction'] = y_pred

    sns.displot(df)
    plt.title(f"Comparing Target and Prediction Distributions - {set}")
    plt.grid(True)
    plt.show()