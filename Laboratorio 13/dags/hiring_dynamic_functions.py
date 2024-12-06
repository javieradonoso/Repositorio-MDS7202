import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# 1. Cargar y unir datos
def load_and_merge(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    raw_path = os.path.join(f'data_{execution_date}', 'raw')
    preprocessed_path = os.path.join(f'data_{execution_date}', 'preprocessed')

    data_files = [os.path.join(raw_path, f) for f in ['data_1.csv', 'data_2.csv'] if os.path.exists(os.path.join(raw_path, f))]

    df = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)
    output_path = os.path.join(preprocessed_path, 'merged_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Archivos unidos y guardados en {output_path}")


# 2. División de datos
def split_data(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    preprocessed_path = os.path.join(f'data_{execution_date}', 'preprocessed', 'merged_data.csv')
    splits_path = os.path.join(f'data_{execution_date}', 'splits')

    df = pd.read_csv(preprocessed_path)
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=99
    )

    X_train.to_csv(os.path.join(splits_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(splits_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(splits_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(splits_path, 'y_test.csv'), index=False)


# 3. Evaluación de modelos
def evaluate_models(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    splits_path = os.path.join(f'data_{execution_date}', 'splits')
    models_path = os.path.join(f'data_{execution_date}', 'models')

    X_test = pd.read_csv(os.path.join(splits_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(splits_path, 'y_test.csv')).squeeze()

    model_files = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
    best_model = None
    best_accuracy = 0

    for model_file in model_files:
        model_path = os.path.join(models_path, model_file)
        clf = joblib.load(model_path)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modelo {model_file}: Accuracy = {acc:.2f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model_file

    if best_model:
        best_model_path = os.path.join(models_path, 'best_model.joblib')
        os.rename(os.path.join(models_path, best_model), best_model_path)
        print(f"Mejor modelo: {best_model} con Accuracy = {best_accuracy:.2f}")


# 4. Entrenamiento del modelo
def train_model(model, **kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    splits_path = os.path.join(f'data_{execution_date}', 'splits')
    models_path = os.path.join(f'data_{execution_date}', 'models')

    X_train = pd.read_csv(os.path.join(splits_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(splits_path, 'y_train.csv')).squeeze()

    numerical_features = ['Age', 'ExperienceYears', 'PreviousCompanies',
                           'DistanceFromCompany', 'InterviewScore',
                           'SkillScore', 'PersonalityScore']
    categorical_features = ['Gender', 'EducationLevel', 'RecruitmentStrategy']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    clf.fit(X_train, y_train)
    model_name = model.__class__.__name__
    model_path = os.path.join(models_path, f'{model_name}_hiring_model.joblib')
    joblib.dump(clf, model_path)
    print(f"Modelo {model_name} entrenado y guardado en {model_path}")


# 5. Crear carpetas
def create_folders(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    folder_name = f"data_{execution_date}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'preprocessed'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'models'), exist_ok=True)
    print(f"Carpetas creadas en {folder_name}")