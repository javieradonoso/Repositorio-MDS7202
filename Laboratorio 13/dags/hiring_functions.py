import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr

# Ruta base del proyecto
BASE_PATH = os.path.join(os.getcwd(), 'dags')

# 1. Función create_folders
def create_folders(base_path=BASE_PATH, execution_date=None):
    execution_date = datetime.now().strftime("%Y-%m-%d")
    main_folder = os.path.join(base_path, execution_date)
    
    # Crear carpeta principal y subcarpetas
    os.makedirs(main_folder, exist_ok=True)
    subfolders = ['raw', 'splits', 'models']
    for subfolder in subfolders:
        os.makedirs(os.path.join(main_folder, subfolder), exist_ok=True)
    
    print(f"Carpetas creadas en: {main_folder}")
    return main_folder

# 2. Función split_data

# Ruta base del proyecto
BASE_PATH = os.path.join(os.getcwd(), 'dags')

def split_data(test_size=0.2, random_state=42):
    # Obtener la fecha actual como execution_date
    execution_date = datetime.now().strftime("%Y-%m-%d")
    
    # Definir rutas basadas en la fecha de ejecución
    raw_folder = os.path.join(BASE_PATH, execution_date, 'raw')
    splits_folder = os.path.join(BASE_PATH, execution_date, 'splits')
    
    # Ruta del archivo raw
    file_path = os.path.join(raw_folder, 'data_1.csv')
    
    # Verificar que el archivo exista
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Dividir los datos
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['HiringDecision']
    )
    
    # Guardar los conjuntos
    train_path = os.path.join(splits_folder, 'train.csv')
    test_path = os.path.join(splits_folder, 'test.csv')
    
    # Crear directorio para splits si no existe
    os.makedirs(splits_folder, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    return train_path, test_path

# 3. Función preprocess_and_train

def preprocess_and_train():
    # Obtener la fecha actual como execution_date
    execution_date = datetime.now().strftime("%Y-%m-%d")
    
    # Definir rutas
    splits_folder = os.path.join(BASE_PATH, execution_date, 'splits')
    models_folder = os.path.join(BASE_PATH, execution_date, 'models')
    
    # Rutas de los archivos de entrenamiento y prueba
    train_path = os.path.join(splits_folder, 'train.csv')
    test_path = os.path.join(splits_folder, 'test.csv')
    
    # Verificar que los archivos existan
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Los archivos de entrenamiento y prueba no existen en la carpeta splits.")
    
    # Cargar los datos
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separar características y etiquetas
    X_train, y_train = train_data.drop(columns='HiringDecision'), train_data['HiringDecision']
    X_test, y_test = test_data.drop(columns='HiringDecision'), test_data['HiringDecision']
    
    # Definir columnas numéricas
    numeric_features = ['Age', 'ExperienceYears', 'DistanceFromCompany', 
                        'InterviewScore', 'SkillScore', 'PersonalityScore']
    
    # Preprocesamiento: escalamiento para variables numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Modelo RandomForest
    model = RandomForestClassifier(random_state=42)
    
    # Pipeline con preprocesamiento y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluación en el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Guardar el modelo entrenado en la carpeta models
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, 'model.joblib')
    joblib.dump(pipeline, model_path)
    
    print(f"Modelo guardado en: {model_path}")
    return model_path


    
    # Verificar que el modelo exista
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encuentra en {model_path}. Asegúrate de entrenarlo antes de usar esta función.")
    

# 4. Función gradio_interface

def gradio_interface():
    # Obtener la fecha actual como execution_date
    execution_date = datetime.now().strftime("%Y-%m-%d")

    # Ruta al modelo guardado
    models_folder = os.path.join(BASE_PATH, execution_date, 'models')
    model_path = os.path.join(models_folder, 'model.joblib')

    # Cargar el modelo
    pipeline = joblib.load(model_path)
    
    # Función de predicción
    def predict_from_json(json_file):
        input_data = pd.read_json(json_file, orient='records')
        # Verificar que las columnas requeridas existan
        expected_columns = [
            'Age', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany',
            'InterviewScore', 'SkillScore', 'PersonalityScore', 'Gender',
            'EducationLevel', 'RecruitmentStrategy'
        ]
        if not all(col in input_data.columns for col in expected_columns):
            return f"El archivo JSON debe contener las columnas: {', '.join(expected_columns)}"
        # Realizar predicciones
        predictions = pipeline.predict(input_data)
        input_data['Prediction'] = predictions
        input_data['Prediction'] = input_data['Prediction'].apply(lambda x: "Contratado" if x == 1 else "No Contratado")
        
        # Devolver los resultados como JSON
        return input_data.to_json(orient='records', lines=True)
    
    # Crear la interfaz de Gradio
    interface = gr.Interface(
        fn=predict_from_json,
        inputs=gr.File(label="Suba un archivo JSON con los datos del candidato"),
        outputs="text",
        title="Hiring Prediction Model",
        description="Suba un archivo JSON con los datos de candidatos para predecir si serán contratados o no."
    )
    
    # Lanzar la interfaz
    interface.launch(share=True)