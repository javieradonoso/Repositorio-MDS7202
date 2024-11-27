# importamos la librerias 
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pkg_resources
import warnings
import os

# colocamos los warnings para que no nos mande al correr el codigo
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Ruta base del archivo actual (donde está optimize.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Crear carpetas necesarias dentro del directorio base
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def get_best_model(experiment_id):
    # Buscar todas las ejecuciones en el experimento dado
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Verificar si hay ejecuciones disponibles
    if runs.empty:
        raise ValueError(f"No se encontraron runs en el experimento con ID {experiment_id}.")
    
    # Ordenar por la métrica 'valid_f1' y seleccionar la mejor ejecución
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.xgboost.load_model(f"runs:/{best_model_id}/model")
    return best_model

# Uso de rutas absolutas en el código
def optimize_model(learning_rate_values):
    # Cargar el dataset
    dataset_path = os.path.join(BASE_DIR, "water_potability.csv")
    df = pd.read_csv(dataset_path)

    # Eliminamos las entradas con valores nulos
    df = df.dropna()

    # Dividir el dataset
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    for learning_rate in learning_rate_values:
        # Configurar experimento de MLflow
        experiment_name = f"XGBoost_LR_{learning_rate}"
        mlflow.set_experiment(experiment_name)

        # Crear un estudio
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            # Finalizar cualquier run activo
            if mlflow.active_run() is not None:
                mlflow.end_run()

            # Definir los hiperparámetros que Optuna utiliza
            param_grid = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': learning_rate,
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            }

            # Crear el modelo con los hiperparámetros sugeridos
            model = xgb.XGBClassifier(**param_grid, random_state=42)

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Realizar predicciones
            y_pred = model.predict(X_test)

            # Calcular el F1-Score
            f1 = f1_score(y_test, y_pred)

            # Registrar en un nuevo run de MLflow
            with mlflow.start_run(run_name=f"XGBoost_Run_LR_{learning_rate:.2f}_Trial_{trial.number}"):
                mlflow.log_params(param_grid)
                mlflow.log_metric("valid_f1", f1)
                mlflow.xgboost.log_model(model, "model")

            return f1  # Optuna buscará maximizar este valor

        # Optimizar hiperparámetros para este learning_rate
        study.optimize(objective, n_trials=25)

        # Generar el gráfico de Optuna (Optimization History)
        fig1 = plot_optimization_history(study)

        # Ajustar el tamaño del gráfico
        fig1.figure.set_size_inches(10, 5)

        # Guardar el gráfico como archivo PNG
        plot_path = os.path.join(PLOTS_DIR, f"optimization_history_lr_{learning_rate}_.png")
        fig1.figure.savefig(plot_path, bbox_inches="tight")
        mlflow.log_artifact(plot_path)

        # Devolver y guardar el mejor modelo para este experimento
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        best_model = get_best_model(experiment_id)

        # Guardar el modelo con pickle
        model_path = os.path.join(MODELS_DIR, f"best_model_lr_{learning_rate}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model.get_booster(), f)

        # Guardar la importancia de las variables
        importance = best_model.get_booster().get_score(importance_type="weight")
        importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Importance"])

        plt.figure(figsize=(12, 6))
        importance_df.sort_values(by="Importance", ascending=False).plot(
            kind="bar",
            x="Feature",
            y="Importance",
            title=f"Feature Importance LR {learning_rate}",
            legend=True,
            figsize=(12, 6)
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Guardar el gráfico
        feature_importance_path = os.path.join(PLOTS_DIR, f"feature_importance_lr_{learning_rate}.png")
        plt.savefig(feature_importance_path)
        mlflow.log_artifact(feature_importance_path)

# Definir el entrypoint del script
if __name__ == "__main__":
    learning_rate_values = [0.3, 0.4, 0.5]
    optimize_model(learning_rate_values)