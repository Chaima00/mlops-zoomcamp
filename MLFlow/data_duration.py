#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
from pathlib import Path
import mlflow.xgboost
import numpy as np  



mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('my-nyc-taxi-experiment')



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df = df[(df.duration >= 1)&(df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df




df_train = read_dataframe(year=2023, month=1)
df_val = read_dataframe(year=2023, month=2)

def create_X(df, dv = None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse= True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv



X_train, dv = create_X(df_train)
X_val, _ = create_X(df_val, dv)

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values





models_folder = Path('models')
models_folder.mkdir(exist_ok= True)



def train(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'pbjective' : 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b', artifact_path='preprocessor')

        mlflow.xgboost.log_model(booster, name='models_mlflow')


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    train(X_train, y_train, X_val, y_val, dv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train XGBoost model for NYC Taxi duration prediction')
    parser.add_argument('--year', type=int, default=2023, help='Year of the training data')
    parser.add_argument('--month', type=int, default=1, help='Month of the training data')
    args = parser.parse_args()
    run(year=args.year, month=args.month)

