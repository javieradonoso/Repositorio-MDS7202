from airflow import DAG
from airflow.operators.empty import EmptyOperator  # Cambiado DummyOperator a EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import os
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Definir el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0
}

BASE_PATH = os.path.join(os.getcwd(), 'dags')

# URL del dataset
DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv" 

with DAG(
    dag_id='hiring_lineal',
    default_args=default_args,
    description='Pipeline para predecir contrataciones',
    schedule=None,  # Usar el nuevo parámetro `schedule`
    start_date=datetime(2024, 10, 1),
    catchup=False,
    tags=['hiring', 'pipeline']
) as dag:

    # Punto de inicio
    start_pipeline = EmptyOperator(  # Cambiado DummyOperator a EmptyOperator
        task_id='start_pipeline'
    )

    # Crear carpetas
    create_folders_task = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={
            'base_path': BASE_PATH,
            'execution_date': '{{ ds }}'
        }
    )

    # Descargar el archivo data_1.csv
    download_data_task = BashOperator(
        task_id='download_data',
        bash_command=f"curl -o {BASE_PATH}/{{{{ ds }}}}/raw/data_1.csv {DATA_URL}"
    )

    # Realizar hold-out con split_data
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={
            'execution_date': '{{ ds }}'
        }
    )

    # Entrenamiento y preprocesamiento
    preprocess_and_train_task = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=preprocess_and_train
    )

    # Lanzar interfaz de Gradio
    gradio_interface_task = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface
    )

    # Definir dependencias del DAG
    start_pipeline >> create_folders_task >> download_data_task >> split_data_task >> preprocess_and_train_task >> gradio_interface_task