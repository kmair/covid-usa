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

# Merging with the state code
states_url = 'https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv'
pop_df = pd.read_csv(states_url)
pop_df.drop(['Rank'], axis=1, inplace=True)
df = df.merge(pop_df.iloc[:,:2], left_on='state', right_on='State') # Previous line can be avoided by selecting iloc[1:3]
df.drop('State', axis=1, inplace=True)

print(df)

intro_markdown = '''
&emsp;This app uses daily data posted by [The New York Times](https://github.com/nytimes/covid-19-data/blob/master/us-states.csv)

&emsp;The tally of Cases and Deaths at present and forecasts based on time-series can be evaluated below
'''

def return_usa_map(df, col):
    fig = px.choropleth(
        df,
        locations='Postal',
        locationmode="USA-states",
        scope='usa',
        color=col,
        hover_name='Postal'
        # ,hover_data=['state']
    )
    fig.update_layout(
        title_text = f'USA {col}'
    )
    return fig

app.layout = html.Div(children=[

    html.H1('Covid in the USA', className="mb-2", style={'text-align': 'center'}),
    dcc.Markdown(children=intro_markdown),
 
    dcc.Tabs(id='cases_or_deaths', value='cases', children=[
            dcc.Tab(label='Cases', value='cases'),
            dcc.Tab(label='Deaths', value='deaths')
        ]),
    html.Div(id='cases_or_deaths_content')

    ],
    
    # style={'background-image': 'url("/assets/corona-background.jpg")', 'background-color': 'rgba(255, 255, 255, 0.16)'}
    style={'background-image': 'url("/assets/coronavirusbg.png")', 'background-size': '1400px 300px'}
)
@app.callback(Output('cases_or_deaths_content', 'children'),
              [Input('cases_or_deaths', 'value')]
)
def render_tabs(tab):
    if tab == 'cases':
        return html.Div(children=[
            dcc.Graph(
            id='usa-map',
            figure=return_usa_map(df, tab)
            )
        ])
    elif tab == 'deaths':
        return html.Div(children=[
            dcc.Graph(
            id='usa-map',
            figure=return_usa_map(df, tab)
            # figure=fig
            )
        ])


# @app.callback(
#     [Output(component_id='usa_map', component_property='figure')],
#     [Input()]
# )

if __name__ == "__main__":
    app.run_server(debug=True)
    # pass