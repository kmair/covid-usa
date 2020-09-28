import pandas as pd 
import numpy as np
import torch
from urllib.request import urlopen
import json

import plotly.express as px
import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
# dash-table

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('covid_state_9_27.csv')

df = df.groupby(['state']).max()
df.reset_index(inplace=True)
print(df)
# geo_fips = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
# with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#     counties = json.load(response)

fig = px.choropleth(
    df,
    locations='state',
    locationmode="USA-states",
    scope='usa',
    color='cases'
    # ,hover_data=['state']
)

app.layout = html.Div([

   html.H1('Covid in the USA', className="mb-2", style={'text-align': 'center'}),

 
    dcc.Graph(
        id='usa_map',
        figure=fig
        )
    #    id='usa_map') 

])
# )
# @app.callback(
#     [Output(component_id='usa_map', component_property='figure')],
#     [Input()]
# )

if __name__ == "__main__":
    app.run_server(debug=True)
    # pass