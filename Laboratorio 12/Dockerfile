# 1. Imagen base de Python
FROM python:3.9-slim

# 2. Crear un directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo requirements.txt para instalar las dependencias
COPY requirements.txt .

# 4. Actualizar pip e instalar dependencias
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto de los archivos del proyecto al contenedor
COPY . .

# 6. Establecer variables de entorno (opcional)
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random

# 7. Exponer el puerto en el que correrá la API (por defecto 8000 para FastAPI)
EXPOSE 8000

# 8. Comando por defecto para iniciar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]