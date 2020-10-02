import pandas as pd 
import numpy as np
import torch
from urllib.request import urlopen
import json
import datetime

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
# dash-table

color_scale = {'cases': 'Blues', 'deaths': 'Reds'} # https://plotly.com/python/builtin-colorscales/

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('covid_state_9_27.csv')


# Merging with the state code
states_url = 'https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv'
pop_df = pd.read_csv(states_url)
pop_df.drop(['Rank'], axis=1, inplace=True)
df = df.merge(pop_df.iloc[:,:2], left_on='state', right_on='State') # Previous line can be avoided by selecting iloc[1:3]
df.drop('State', axis=1, inplace=True)


intro_markdown = '''
&emsp;This app uses daily data posted by [The New York Times](https://github.com/nytimes/covid-19-data/blob/master/us-states.csv)

&emsp;The tally of Cases and Deaths at present and forecasts based on time-series can be evaluated below
'''

def return_usa_plots(df, column):

    # Latest total
    state_df = df.groupby(['state']).max()
    state_df.reset_index(inplace=True)

    fig_usa_map = px.choropleth(
        state_df,
        locations='Postal',
        locationmode="USA-states",
        scope='usa',
        color=column,
        color_continuous_scale=color_scale[column],
        hover_name='Postal'
        # ,hover_data=['state']
    )
    fig_usa_map.update_layout(
        title_text = f'USA {column}'
    )

    # Plot total cases/deaths by date
    date_df = df.groupby('date').sum()

    fig_usa_total = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_usa_total.add_trace(
        go.Scatter(
        x = date_df.index,
        y = date_df[column].diff(),
        name=f"Daily {column}",
        fill='tozeroy'
        ), secondary_y = True
    )
    
    fig_usa_total.add_trace(
        go.Scatter(
        x = date_df.index,
        y = date_df[column],
        name=f"Total {column}",
        
        # labels=dict(x="", y=f"Total {column}")
        ), secondary_y = False
    )

    fig_usa_total.update_layout(xaxis_range=[datetime.datetime(2020, 1, 10),
                               datetime.date.today() + datetime.timedelta(days=10)])

    return fig_usa_map, fig_usa_total

def return_usa_stats(df, column):
    new_df = df.loc[:, ['date', 'state', column]].set_index('date')
    state_df = new_df.groupby('state')

    # Total and % of total affected
    latest_df = state_df.agg(['last'])
    latest_df = latest_df.droplevel(1, axis=1)  # removing agg of last from column
    latest_df = pop_df.merge(latest_df, left_on='State', right_index=True).drop(['Postal'], axis=1)
    # print(latest_df)
    latest_df['Total per capita (%)'] = latest_df[column]/latest_df['Population']*100
    latest_df.rename({column: f'Total {column}'}, axis = 1, inplace=True)
    
    # Growth over the past week
    # rolling_sum = state_df.rolling(5).sum()  # Past week count
    # rolling_sum = rolling_sum.reset_index().groupby('state').agg('last')
    # rolling_sum = rolling_sum.drop('date', axis=1).rename({column: f'Increase in a week'}, axis=1)
    weekly_increase = state_df.apply(lambda x: (x[column].iloc[-1]-x[column].iloc[-8]))
    weekly_increase = pd.DataFrame(data=weekly_increase, columns=['Increase in a week'])
    df = latest_df.round(decimals=3).merge(weekly_increase, left_on='State', right_index=True).drop(['Population'], axis=1)

    # Plotly
    stats_table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        filter_action="native",
        sort_action="native",
        row_selectable="multi", 
        column_selectable="single",
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0, 
        page_size=6,
        style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in {'State'}
        ],
        style_data={
            'whitespace': 'normal',
            'height': 'auto'
        }
    )

    return stats_table

app.layout = html.Div(children=[
    html.Div(children=[        
        html.Br(),
        html.H1('Covid in the USA', className="mb-2", style={'text-align': 'center'}),
        dcc.Markdown(children=intro_markdown),
        html.Br()
        ]
    ),
    dcc.Tabs(id='cases_or_deaths', value='cases', children=[
            dcc.Tab(label='Cases', value='cases'),
            dcc.Tab(label='Deaths', value='deaths')
        ]),
    html.Div(id='cases_or_deaths_content'),
    html.Br(),
    html.H1('Analysis of the state USA', className="mb-2", style={'text-align': 'center'}),
                
    ],
    
    # style={'background-image': 'url("/assets/corona-background.jpg")', 'background-color': 'rgba(255, 255, 255, 0.16)'}
    style={'background-image': 'url("/assets/coronavirusbg.png")', 'background-size': '1400px 300px'}
)

@app.callback(Output('cases_or_deaths_content', 'children'),
              [Input('cases_or_deaths', 'value')]
)
def render_tabs(tab):
    usa_map, usa_total = return_usa_plots(df, tab)

    stats_table = return_usa_stats(df, tab)
    # print(table)

    usa_plots = html.Div([
        html.Div(
        children=[
            html.Div(children=[
                    dcc.Graph(
                    id='usa-map',
                    figure=usa_map   
                    )
                ], className='col-md-6 px-0'
            ),
            html.Div(children=[
                    dcc.Graph(
                    id='usa-map',
                    figure=usa_total # return_usa_plot(df, tab)
                    )
                ], className='col-md-6 px-0'
            )
        ], className='row mx-0'),

        html.Div(
        children=[
            html.Div(children=[stats_table], className='col-md-6'
            ),
            html.Div(children=[
                    dcc.Graph(
                    id='usa-map',
                    figure=usa_total 
                    )
                ], className='col-md-6 px-0'
            )
        ], className='row mx-0')

    ]
    )
    return usa_plots
    

if __name__ == "__main__":
    app.run_server(debug=True)