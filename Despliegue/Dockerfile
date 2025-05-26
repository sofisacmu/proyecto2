# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /app

# Copiar archivos al contenedor
COPY Dash.py modelo.keras info_modelo.pkl requirements.txt /app/

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto correcto
EXPOSE 8050

# Ejecutar la app
CMD ["python", "Dash.py"]