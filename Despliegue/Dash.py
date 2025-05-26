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
df = pd.read_csv("https://www.dropbox.com/scl/fi/6q2synntpk2v5ng7w47rg/Datos2011Limipios.csv?rlkey=m1i9eo8sur1azd13swvqy1teh&st=z00mielw&dl=1")
modelo = tf.keras.models.load_model("modelo.keras")

# Cargar información de preprocesamiento
with open("info_modelo.pkl", "rb") as f:
    prep_info = joblib.load(f)

# Extraer columnas esperadas y codificadores
categorical_columns = prep_info['categorical_columns']
feature_columns = prep_info['feature_columns']
encoders = prep_info['encoders']

# Extraer categorías directamente desde los encoders
def get_categories(prefix):
    return list({val for col, le in encoders.items() if col.startswith(prefix) for val in le.classes_})

departamentos = get_categories("cole_depto_ubicacion")
jornadas = get_categories("cole_jornada")
nats = get_categories("cole_naturaleza")
mcpios = get_categories("cole_mcpio_ubicacion")
ed_madre = get_categories("fami_educacionmadre")
ed_padre = get_categories("fami_educacionpadre")
estratos = get_categories("fami_estratovivienda")

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

# Nombres de facil comprension para el usuario
nombres = {
    'cole_depto_ubicacion': 'Departamento del colegio',
    'cole_jornada': 'Jornada escolar',
    'cole_mcpio_ubicacion': 'Municipio del colegio',
    'cole_naturaleza': 'Tipo de institución',
    'fami_educacionmadre': 'Nivel educativo madre',
    'fami_educacionpadre': 'Nivel educativo padre',
    'fami_estratovivienda': 'Estrato socioeconómico'
}

# Inicializar Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Simulador Saber 11 - Inglés"

# Layout
app.layout = dbc.Container([
    html.H2("Simulador de Desempeño en Inglés en las Pruebas Saber 11", className="text-center mt-4"),
    html.P("Bienvenido, si eres estudiante, con este simulador podrás estimar tu perfil de desempeño en la prueba de inglés del examen Saber 11 según tus características",
           className="lead text-center"),
    html.Hr(),

    dbc.Alert([
        html.H5("¿Cómo usar el simulador?", className="alert-heading"),
        html.Ul([
            html.Li([
                html.Strong("1. Analiza los patrones de desempeño: "),
                "Explora las gráficas superiores que muestran las características comunes de estudiantes con alto desempeño en inglés. Observa cómo factores como el estrato socioeconómico y el tipo de colegio se relacionan con los resultados."
            ], className="mb-2"),
            html.Li([
                html.Strong("2. Completa tu información personal: "),
                "En la sección 'Simula tu perfil', selecciona todas las opciones que correspondan a tu situación actual: departamento, municipio, tipo de institución educativa, jornada escolar, nivel educativo de tus padres y estrato socioeconómico."
            ], className="mb-2"),
            html.Li([
                html.Strong("3. Obtén tu evaluación personalizada: "),
                "Al hacer clic en 'Evaluar mi perfil', recibirás una predicción basada en modelos estadísticos que compara tu perfil con los patrones de estudiantes exitosos. El sistema te indicará si tu perfil se asemeja más a un desempeño ",
                html.Strong("exitoso", className="text-success"),
                " o ",
                html.Strong("no exitoso", className="text-danger"),
                "."
            ], className="mb-2"),
        ])
    ], color="info"),

    html.Hr(),
    html.H4("Características de estudiantes con mejores resultados", className="mt-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(
            figure=px.bar(
                df[df["resultado"] == "exitoso"]["fami_estratovivienda"].value_counts().reset_index(),
                x="fami_estratovivienda",
                y="count",
                title="Número de estudiantes exitosos por estrato socioeconómico",
                labels={'fami_estratovivienda': 'Estrato', 'count': 'Número de estudiantes'},
                color="fami_estratovivienda"
            ).update_layout(showlegend=False)
        ), width=5),

        dbc.Col(dcc.Graph(
            figure=px.density_heatmap(
                df[df["resultado"] == "exitoso"],
                x="fami_educacionmadre",
                y="fami_educacionpadre",
                z="punt_ingles",
                histfunc="avg",
                title="Puntaje de inglés según educación de los padres"
            ).update_layout(
                xaxis_title="Educación de la madre",
                yaxis_title="Educación del padre",
                plot_bgcolor='white'
            ).update_coloraxes(colorbar_title="Promedio puntaje de inglés")
        ), width=7)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(
            figure=px.box(
                df[df["resultado"] == "exitoso"],
                x="cole_depto_ubicacion",
                y="punt_ingles",
                title="Distribución de puntajes por departamento",
                labels={'cole_depto_ubicacion': 'Departamento', 'punt_ingles': 'Puntaje inglés'}
            ).update_layout(
                showlegend=False,
                xaxis={'categoryorder': 'total descending', 'tickangle': 45},
                plot_bgcolor='white',
                margin=dict(b=120)
            )
        ), width=12)
    ], className="mb-4"),

    html.H5("Simula tu perfil", className="mt-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['cole_depto_ubicacion']),
            dcc.Dropdown(id='input-cole_depto_ubicacion',
                         options=[{'label': d, 'value': d} for d in sorted(departamentos)],
                         placeholder="Selecciona departamento")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['cole_jornada']),
            dcc.Dropdown(id='input-cole_jornada',
                         options=[{'label': j, 'value': j} for j in sorted(jornadas)],
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
                         options=[{'label': n, 'value': n} for n in sorted(nats)],
                         placeholder="Selecciona tipo de institución")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['fami_educacionmadre']),
            dcc.Dropdown(id='input-fami_educacionmadre',
                         options=[{'label': e, 'value': e} for e in sorted(ed_madre)],
                         placeholder="Selecciona educación madre")
        ], width=6),
        dbc.Col([
            dbc.Label(nombres['fami_educacionpadre']),
            dcc.Dropdown(id='input-fami_educacionpadre',
                         options=[{'label': e, 'value': e} for e in sorted(ed_padre)],
                         placeholder="Selecciona educación padre")
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label(nombres['fami_estratovivienda']),
            dcc.Dropdown(id='input-fami_estratovivienda',
                         options=[{'label': e, 'value': e} for e in sorted(estratos)],
                         placeholder="Selecciona estrato")
        ], width=6),
    ], className="mb-3"),

    dbc.Button("Evaluar mi perfil", id="btn-evaluar", color="primary", className="mb-3"),
    html.Div(id="resultado-evaluacion", className="h4 text-center", style={"marginTop": "20px"}),

], fluid=True)

# Callback para que la lista de municipios dependa del departamento seleccionad
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

# Callback para evaluar el perfil
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

    try:
        # Codificar variables categóricas
        for col in categorical_columns:
            if col in df_input.columns:
                df_input[col] = df_input[col].fillna('unknown')
                le = encoders[col]
                df_input[col] = df_input[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df_input[col] = le.transform(df_input[col])
        
        # Normalizar
        X_scaled = prep_info['scaler'].transform(df_input[feature_columns])

        # Predecir
        prob = modelo.predict(X_scaled, verbose=0)[0][0]
        if prob > 0.5:
            resultado = "EXITOSO"
            prob_mostrar = prob
            color = "success"
        else:
            resultado = "NO EXITOSO"
            prob_mostrar = 1 - prob
            color = "danger"

        porcentaje = round(prob_mostrar * 100)

        return dbc.Alert(f"Tu perfil fue clasificado como: {resultado} con una probabilidad del {porcentaje}% en la prueba de inglés", color=color)


    except Exception as e:
        return dbc.Alert(f"Error al predecir: {str(e)}", color="danger")

# Ejecutar servidor
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
