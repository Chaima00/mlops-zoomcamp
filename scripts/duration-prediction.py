#!/usr/bin/env python
# coding: utf-8





import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from pathlib import Path


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def feature_engineering(df, dv = None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
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
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year  if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = feature_engineering(df_train)
    X_val, _ = feature_engineering(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    train_model(X_train, y_train, X_val, y_val, dv)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for NYC taxi duration prediction")
    parser.add_argument('--year', type=int, default=2023, help='Year of the data to train on')
    parser.add_argument('--month', type=int, default=3, help='Month of the data to train on')
    args = parser.parse_args()  
    run(year=args.year, month=args.month) 