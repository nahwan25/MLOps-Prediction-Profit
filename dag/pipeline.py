from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='training_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    train = BashOperator(
        task_id='train_model',
        bash_command='python src/train.py',        # jalankan script di dalam src/
        cwd='/opt/airflow/dags'                    # working directory = folder dag
    )

    evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command='python src/evaluate.py',
        cwd='/opt/airflow/dags'
    )

    train >> evaluate