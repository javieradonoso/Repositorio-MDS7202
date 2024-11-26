from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from xgboost import DMatrix

# Cargar el modelo previamente guardado
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Potabilidad del Agua",
    description="Esta API predice si una medición de agua es potable o no utilizando un modelo optimizado de XGBoost.",
    version="1.0.0"
)

# Modelo de datos esperado para las solicitudes POST
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta GET para descripción del modelo
@app.get("/")
def read_root():
    return {
        "message": "API para predicción de potabilidad del agua.",
        "usage": "Use el endpoint /potabilidad/ con un POST para predecir.",
        "docs": "Visite /docs para más detalles."
    }

# Ruta POST para predecir la potabilidad
@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    try:
        # Convertir los datos de entrada en un DataFrame
        data = pd.DataFrame([sample.dict()])

        # Si el modelo requiere un DMatrix, conviértelo
        dmatrix_data = DMatrix(data)

        # Realizar la predicción
        prediction = model.predict(dmatrix_data)

        # Convertir la predicción a un valor binario
        prediction_binary = int(prediction[0] > 0.5)

        # Retornar el resultado
        return {"potabilidad": prediction_binary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando los datos: {str(e)}")