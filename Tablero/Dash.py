# Importar librerías
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import tensorflow as tf
import numpy as np
import joblib

# Cargar datos y modelo 
df = pd.read_csv("/Users/valeriacardenas/Desktop/proyecto2/Tablero/Datos2011Limipios.csv")
modelo = tf.keras.models.load_model("/Users/valeriacardenas/Desktop/proyecto2/Modelamiento/modelos_guardados/mejor_modelo.keras")

# Cargar columnas esperadas desde archivo pkl
columnas_esperadas = joblib.load("/Users/valeriacardenas/Desktop/proyecto2/Modelamiento/modelos_guardados/columns_info_limpio.pkl")
if isinstance(columnas_esperadas, dict):
    columnas_esperadas = list(columnas_esperadas.keys())

# Extraer categorías directamente de las columnas esperadas 
deptos = [col.replace("cole_depto_ubicacion_", "") for col in columnas_esperadas if col.startswith("cole_depto_ubicacion_")]
jornadas = [col.replace("cole_jornada_", "") for col in columnas_esperadas if col.startswith("cole_jornada_")]
nats = [col.replace("cole_naturaleza_", "") for col in columnas_esperadas if col.startswith("cole_naturaleza_")]
mcpios = [col.replace("cole_mcpio_ubicacion_", "") for col in columnas_esperadas if col.startswith("cole_mcpio_ubicacion_")]
ed_madre = [col.replace("fami_educacionmadre_", "") for col in columnas_esperadas if col.startswith("fami_educacionmadre_")]
ed_padre = [col.replace("fami_educacionpadre_", "") for col in columnas_esperadas if col.startswith("fami_educacionpadre_")]
estratos = [col.replace("fami_estratovivienda_", "") for col in columnas_esperadas if col.startswith("fami_estratovivienda_")]

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

# Nombres de facil comprension
nombres = {
    'cole_depto_ubicacion': 'Departamento del colegio',
    'cole_jornada': 'Jornada escolar',
    'cole_mcpio_ubicacion': 'Municipio del colegio',
    'cole_naturaleza': 'Tipo de institución',
    'fami_educacionmadre': 'Nivel educativo madre',
    'fami_educacionpadre': 'Nivel educativo padre',
    'fami_estratovivienda': 'Estrato socioeconómico'
}

# Inicializar dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Simulador Saber 11 - Inglés"

# Layout del dash
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

    html.H4("Simula tu perfil", className="mt-4"),

    # Inputs del usuario
    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['cole_depto_ubicacion']),
            dcc.Dropdown(id='input-cole_depto_ubicacion',
                         options=[{'label': d, 'value': d} for d in sorted(set(deptos))],
                         placeholder="Selecciona departamento")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['cole_jornada']),
            dcc.Dropdown(id='input-cole_jornada',
                         options=[{'label': j, 'value': j} for j in sorted(set(jornadas))],
                         placeholder="Selecciona jornada")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['cole_mcpio_ubicacion']),
            dcc.Dropdown(id='input-cole_mcpio_ubicacion',
                        options=[],
                        placeholder="Selecciona municipio según departamento",
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
                         options=[{'label': e, 'value': e} for e in sorted(set(ed_madre))],
                         placeholder="Selecciona educación madre")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['fami_educacionpadre']),
            dcc.Dropdown(id='input-fami_educacionpadre',
                         options=[{'label': e, 'value': e} for e in sorted(set(ed_padre))],
                         placeholder="Selecciona educación padre")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['fami_estratovivienda']),
            dcc.Dropdown(id='input-fami_estratovivienda',
                         options=[{'label': e, 'value': e} for e in sorted(set(estratos))],
                         placeholder="Selecciona estrato")
        ], width=6),
    ], className="mb-3"),

    dbc.Button("Evaluar mi perfil", id="btn-evaluar", color="primary", className="mb-3"),
    html.Div(id="resultado-evaluacion", className="h4 text-center", style={"marginTop": "20px"}),

], fluid=True)


@app.callback(
    Output('input-cole_mcpio_ubicacion', 'options'),
    Output('input-cole_mcpio_ubicacion', 'disabled'),
    Input('input-cole_depto_ubicacion', 'value')
)
def actualizar_municipios(depto):
    if not depto:
        return [], True
    municipios = df[df['cole_depto_ubicacion'] == depto]['cole_mcpio_ubicacion'].dropna().unique()
    return [{'label': m, 'value': m} for m in sorted(municipios)], False



#evaluar
@app.callback(
    Output("resultado-evaluacion", "children"),
    Input("btn-evaluar", "n_clicks"),
    [State(f"input-{col}", "value") for col in features]
)
def evaluar_perfil(n_clicks, *inputs):
    if not n_clicks:
        return ""
    
    if None in inputs:
        return dbc.Alert("Por favor completa todos los campos.", color="warning")

    entrada = dict(zip(features, inputs))
    df_input = pd.DataFrame([entrada])

    # Codificar todas las variables categóricas
    df_dummies = pd.get_dummies(df_input)

    # Crear DataFrame con todas las columnas esperadas en ceros
    df_final = pd.DataFrame(columns=columnas_esperadas)
    df_final.loc[0] = 0

    # Solo copiar columnas que existen en df_final
    for col in df_dummies.columns:
        if col in df_final.columns:
            df_final.at[0, col] = float(df_dummies.at[0, col])

    X_input = df_final.astype(np.float32).values

    try:
        pred = modelo.predict(X_input)[0][0]
        resultado = "EXITOSO" if pred > 0.5 else "NO EXITOSO"
        probabilidad = f"{pred*100:.1f}%" if pred > 0.5 else f"{(1 - pred)*100:.1f}%"
        color = "success" if pred > 0.5 else "danger"
        return dbc.Alert(f"Tu perfil fue clasificado como: {resultado} (Probabilidad: {probabilidad})", color=color)
    except Exception as e:
        return dbc.Alert(f"Error al predecir: {str(e)}", color="danger")


# ejecutar
if __name__ == "__main__":
    app.run_server(debug=True)