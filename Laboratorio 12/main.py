from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import os

# Cargar el modelo previamente guardado
models_dir = 'models'
model_file = None
for filename in os.listdir(models_dir):
    if filename.startswith("best_model_lr_") and filename.endswith(".pkl"):
        model_file = os.path.join(models_dir, filename)
        break

if not model_file:
    raise FileNotFoundError(f"No se encontró un archivo de modelo en la carpeta {models_dir}")

with open(model_file, 'rb') as f:
    best_model = pickle.load(f)

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Potabilidad del Agua",
    description="Esta API predice si una medición de agua es potable o no utilizando un modelo optimizado de XGBoost.",
    version="1.0.0"
)

class WaterSample(BaseModel):
    ph: float = Field(..., ge=0, le=14, description="pH del agua (0-14)")
    Hardness: float = Field(..., ge=0, description="Dureza del agua")
    Solids: float = Field(..., ge=0, description="Sólidos disueltos (mg/L)")
    Chloramines: float = Field(..., ge=0, description="Concentración de cloraminas (mg/L)")
    Sulfate: float = Field(..., ge=0, description="Concentración de sulfato (mg/L)")
    Conductivity: float = Field(..., ge=0, description="Conductividad eléctrica (μS/cm)")
    Organic_carbon: float = Field(..., ge=0, description="Carbono orgánico total (mg/L)")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalometanos (μg/L)")
    Turbidity: float = Field(..., ge=0, description="Turbidez del agua (NTU)")

@app.get("/")
def read_root():
    return {
        "message": "API para predicción de potabilidad del agua.",
        "usage": "Use el endpoint /potabilidad/ con un POST para predecir.",
        "docs": "Visite /docs para más detalles."
    }

@app.get("/modelo/")
def get_model_info():
    return {"modelo_cargado": model_file}

@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    try:
        data = pd.DataFrame([sample.dict()])
        prediction = best_model.predict(data)
        prediction_binary = int(prediction[0] > 0.5)
        return {"potabilidad": prediction_binary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")