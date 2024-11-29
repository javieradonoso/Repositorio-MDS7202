from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Cargar el modelo optimizado desde el archivo
model_path = "models/best_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Crear la aplicación FastAPI
app = FastAPI()

# Clase para validar la entrada del usuario
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

# Ruta principal para describir la API
@app.get("/")
def home():
    return {
        "message": "Bienvenido a la API de Predicción de Potabilidad del Agua",
        "description": "Use la ruta POST /potabilidad para predecir si el agua es potable o no.",
        "input_format": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float",
        },
        "output_format": {
            "potabilidad": "0 o 1, donde 0 indica no potable y 1 indica potable",
        },
    }

# Ruta POST para realizar la predicción
@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    try:
        # Convertir los datos de entrada a un formato compatible con el modelo
        data = np.array([[sample.ph, sample.Hardness, sample.Solids, sample.Chloramines,
                          sample.Sulfate, sample.Conductivity, sample.Organic_carbon,
                          sample.Trihalomethanes, sample.Turbidity]])

        # Realizar la predicción
        prediction = model.predict(data)[0]

        # Retornar el resultado
        return {"potabilidad": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))