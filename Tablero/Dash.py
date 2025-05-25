# Importar librerías
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import tensorflow as tf
import numpy as np
import joblib

# === Cargar datos y modelo ===
df = pd.read_csv("/Users/valeriacardenas/Desktop/proyecto2/Tablero/Datos2011Limipios.csv")
modelo = tf.keras.models.load_model("/Users/valeriacardenas/Desktop/proyecto2/Modelamiento/modelos_guardados/mejor_modelo.keras")

# Cargar columnas esperadas desde archivo pkl
columnas_esperadas = joblib.load("/Users/valeriacardenas/Desktop/proyecto2/Modelamiento/modelos_guardados/columns_info_limpio.pkl")
if isinstance(columnas_esperadas, dict):
    columnas_esperadas = list(columnas_esperadas.keys())

# Variables seleccionadas
features = [
    'cole_depto_ubicacion',
    'cole_jornada',
    'cole_mcpio_ubicacion',
    'cole_naturaleza',
    'fami_educacionmadre',
    'fami_educacionpadre',
    'fami_estratovivienda'
]

# Etiquetas 
nombres = {
    'cole_depto_ubicacion': 'Departamento del colegio',
    'cole_jornada': 'Jornada escolar',
    'cole_mcpio_ubicacion': 'Municipio del colegio',
    'cole_naturaleza': 'Tipo de institución',
    'fami_educacionmadre': 'Nivel educativo madre',
    'fami_educacionpadre': 'Nivel educativo padre',
    'fami_estratovivienda': 'Estrato socioeconómico'
}

# inicializar el dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Simulador Saber 11 - Inglés"

# layout del dash
app.layout = dbc.Container([
    html.H2("Simulador de Desempeño en Inglés - Pruebas Saber 11", className="text-center mt-4"),
    html.P("Este tablero te permite estimar tu perfil de desempeño en la prueba de inglés del examen Saber 11.",
           className="lead text-center"),
    html.Hr(),

    dbc.Alert([
        html.H5("Instrucciones:", className="alert-heading"),
        html.Ul([
            html.Li("1. Observa las gráficas sobre estudiantes exitosos."),
            html.Li("2. Ingresa tus características en el simulador."),
            html.Li("3. Conoce si tu perfil se asemeja a un desempeño exitoso."),
        ])
    ], color="info"),

    html.Hr(),

    html.H4("Características de estudiantes con mejores resultados"),
    
    html.Hr(),

    html.H4("Simula tu perfil", className="mt-4"),

    # Inputs del usuario
    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['cole_depto_ubicacion']),
            dcc.Dropdown(id='input-cole_depto_ubicacion',
                         options=[{'label': d, 'value': d} for d in sorted(df['cole_depto_ubicacion'].dropna().unique())],
                         placeholder="Selecciona departamento")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['cole_jornada']),
            dcc.Dropdown(id='input-cole_jornada',
                         options=[{'label': j, 'value': j} for j in sorted(df['cole_jornada'].dropna().unique())],
                         placeholder="Selecciona jornada")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['cole_mcpio_ubicacion']),
            dcc.Dropdown(id='input-cole_mcpio_ubicacion',
                         placeholder="Primero selecciona un departamento",
                         disabled=True)
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['cole_naturaleza']),
            dcc.Dropdown(id='input-cole_naturaleza',
                         options=[{'label': n, 'value': n} for n in sorted(df['cole_naturaleza'].dropna().unique())],
                         placeholder="Selecciona tipo de institución")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['fami_educacionmadre']),
            dcc.Dropdown(id='input-fami_educacionmadre',
                         options=[{'label': e, 'value': e} for e in sorted(df['fami_educacionmadre'].dropna().unique())],
                         placeholder="Selecciona educación madre")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['fami_educacionpadre']),
            dcc.Dropdown(id='input-fami_educacionpadre',
                         options=[{'label': e, 'value': e} for e in sorted(df['fami_educacionpadre'].dropna().unique())],
                         placeholder="Selecciona educación padre")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['fami_estratovivienda']),
            dcc.Dropdown(id='input-fami_estratovivienda',
                         options=[{'label': e, 'value': e} for e in sorted(df['fami_estratovivienda'].dropna().unique())],
                         placeholder="Selecciona estrato")
        ], width=6),
    ], className="mb-3"),

    dbc.Button("Evaluar mi perfil", id="btn-evaluar", color="primary", className="mb-3"),

    html.Div(id="resultado-evaluacion", className="h4 text-center", style={"marginTop": "20px"}),

], fluid=True)

# Ejecutar
if __name__ == "__main__":
    app.run_server(debug=True)
