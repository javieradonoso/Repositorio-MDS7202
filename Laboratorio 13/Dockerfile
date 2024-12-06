# Usar una imagen base de Python
FROM python:3.10-slim

# Configuración básica
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/dags"

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y curl && apt-get clean

# Configurar directorio de trabajo
WORKDIR /airflow

# Copiar los scripts del proyecto
COPY dags/ /airflow/dags/

# Instalar dependencias compatibles
RUN pip install \
    apache-airflow==2.6.2 \
    numpy==1.22.4 \
    pandas==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.3.2 \
    gradio==3.34.0 \
    pendulum==2.1.2 \
    flask-session==0.4.0 \
    connexion==2.14.1 \
    "pydantic<2.0"\
    xgboost==1.7.6 

# Configuración para Airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/airflow/dags

# Exponer el puerto 8080
EXPOSE 8080

# Comando para iniciar Airflow
CMD ["airflow", "standalone"]