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

# Cambiar las etiquetas de los features selesccionados para que 
# se muestren en el dash de manera mas amigable para el usuario
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

# Callback para la actualización de municipio por departamento
@app.callback(
    Output('input-cole_mcpio_ubicacion', 'options'),
    Output('input-cole_mcpio_ubicacion', 'disabled'),
    Input('input-cole_depto_ubicacion', 'value')
)
def update_municipios(depto):
    if not depto:
        return [], True
    municipios = df[df['cole_depto_ubicacion'] == depto]['cole_mcpio_ubicacion'].dropna().unique()
    return [{'label': m, 'value': m} for m in sorted(municipios)], False

#Callback para la evaluación del perfil
@app.callback(
    Output("resultado-evaluacion", "children"),
    Input("btn-evaluar", "n_clicks"),
    [State(f"input-{col}", "value") for col in features]
)
def evaluar_perfil(n_clicks, *inputs):
    if not n_clicks:
        return ""
    if None in inputs:
        return "Por favor completa todos los campos para continuar."

    entrada = dict(zip(features, inputs))
    df_input = pd.DataFrame([entrada])
    df_dummies = pd.get_dummies(df_input)

    # Asegurar las columnas esperadas
    for col in columnas_esperadas:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
    df_dummies = df_dummies[columnas_esperadas]

    X_input = df_dummies.astype(np.float32).values

    try:
        pred = modelo.predict(X_input)[0][0]
        resultado = "EXITOSO" if pred > 0.5 else "NO EXITOSO"
        probabilidad = f"{pred*100:.1f}%" if pred > 0.5 else f"{(1 - pred)*100:.1f}%"
        return f"🎯 Tu perfil fue clasificado como: **{resultado}** con una probabilidad de {probabilidad}"
    except Exception as e:
        return f"Error al hacer la predicción: {str(e)}"


# Ejecutar
if __name__ == "__main__":
    app.run_server(debug=True)
