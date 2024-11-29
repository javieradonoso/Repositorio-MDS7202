import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pickle
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from sklearn.preprocessing import StandardScaler

# Cargar la base de datos
df = pd.read_csv("water_potability.csv")
df.dropna(inplace=True)  # Eliminar filas con valores faltantes

# Dividir características y etiquetas
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Estandarizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

def objective(trial, X_train, y_train, X_valid, y_valid):
    """Función objetivo para Optuna."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "objective": "binary:logistic",
        "random_state": 42
    }

    # Entrenar modelo
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Calcular F1-Score
    f1 = f1_score(y_valid, y_pred, average="weighted")
    return f1

def optimize_model(X, y):
    """Optimización de modelo con Optuna y registro en MLflow."""
    # Dividir datos en entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Configurar experimento en MLflow
    experiment_name = "XGBoost_Optimization"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Optuna_Optimization"):
        # Registrar versiones de las librerías
        mlflow.log_param("xgboost_version", xgb.__version__)
        mlflow.log_param("optuna_version", optuna.__version__)

        # Crear y ejecutar el estudio de Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=50)

        # Obtener el mejor modelo
        best_params = study.best_params
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)

        # Evaluar y registrar métricas
        y_pred = best_model.predict(X_valid)
        valid_f1 = f1_score(y_valid, y_pred, average="weighted")
        mlflow.log_metric("valid_f1", valid_f1)
        mlflow.log_params(best_params)

        # Guardar el modelo como pickle
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path)

        # Guardar gráficos de Optuna
        os.makedirs("plots", exist_ok=True)
        plot_optimization_history(study).write_image("plots/optimization_history.png")
        plot_param_importances(study).write_image("plots/param_importances.png")

        # Registrar gráficos en MLflow
        mlflow.log_artifact("plots/optimization_history.png")
        mlflow.log_artifact("plots/param_importances.png")

    print(f"Optimización completada. Modelo guardado en {model_path}.")

def get_best_model(experiment_id):
    """
    Obtiene el mejor modelo registrado en un experimento de MLflow
    basado en la métrica valid_f1.

    Parámetros:
    ------------
    experiment_id: str
        El ID del experimento registrado en MLflow.

    Retorno:
    ------------
    best_model:
        Modelo entrenado que obtuvo el mejor valid_f1.
    """
    # Buscar todos los runs del experimento
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Ordenar por la métrica valid_f1 (de mayor a menor)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    
    # Cargar el modelo asociado al mejor run
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    
    return best_model

# Ejecutar la optimización
if __name__ == "__main__":
    optimize_model(X, y)