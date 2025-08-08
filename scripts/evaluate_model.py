from download_data import read_dataframe
from preprocess_data import feature_engineering
from train_model import train_model
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


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