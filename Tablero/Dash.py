#Dash

#Importar librerias
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import tensorflow as tf


# Cargar los datos y el modelo
df = pd.read_csv("/Users/valeriacardenas/Desktop/proyecto2/Tablero/Datos2011Limipios.csv")
modelo = tf.keras.models.load_model("/Users/valeriacardenas/Desktop/proyecto2/Modelamiento/modelo_ej.keras")
