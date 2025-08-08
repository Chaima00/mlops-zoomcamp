from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG('ml_training_pipeline',
         start_date=datetime(2023, 1, 1),
         schedule_interval=None,
         catchup=False) as dag:

    download = BashOperator(
        task_id='download_data',
        bash_command='python scripts/download_data.py'
    )

    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command='python scripts/preprocess_data.py'
    )

    train = BashOperator(
        task_id='train_model',
        bash_command='python scripts/train_model.py'
    )

    evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command='python scripts/evaluate_model.py'
    )

    download >> preprocess >> train >> evaluate
