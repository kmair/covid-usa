import pandas as pd 
import numpy as np
import torch

import plotly.express as px
import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# dash-table

app = dash.Dash(__name__)

df = pd.read_csv('covid_state_9_27.csv')

df = df.groupby(['state'])