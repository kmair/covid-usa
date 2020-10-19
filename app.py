import pandas as pd 
import numpy as np
import torch
from urllib.request import urlopen
import json
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pmdarima import model_selection

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import warnings
warnings.filterwarnings('ignore')
# dash-table

color_scale = {'cases': 'Blues', 'deaths': 'Reds'} # https://plotly.com/python/builtin-colorscales/

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('covid_state_9_27.csv')
df['Death to Case ratio'] = df.deaths / (df.cases + 1e-3) 

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

def return_usa_stats(df, column):
    new_df = df.loc[:, ['date', 'state', 'Death to Case ratio', column]].set_index('date')
    state_df = new_df.groupby('state')

    # Total and % of total affected
    latest_df = state_df.agg(['last'])
    latest_df = latest_df.droplevel(1, axis=1)  # removing agg of last from column
    latest_df = pop_df.merge(latest_df, left_on='State', right_index=True).drop(['Postal'], axis=1)
    latest_df['Total per capita (%)'] = latest_df[column]/latest_df['Population']*100

    latest_df.rename({column: f'Total {column}'}, axis = 1, inplace=True)
    
    # Growth over the past week
    # rolling_sum = state_df.rolling(5).sum()  # Past week count
    # rolling_sum = rolling_sum.reset_index().groupby('state').agg('last')
    # rolling_sum = rolling_sum.drop('date', axis=1).rename({column: f'Increase in a week'}, axis=1)
    weekly_increase = state_df.apply(lambda x: (x[column].iloc[-1]-x[column].iloc[-8]))
    weekly_increase = pd.DataFrame(data=weekly_increase, columns=[f'Weekly {column} toll'])

    df = latest_df.round(decimals=3).merge(weekly_increase, left_on='State', right_index=True).drop(['Population'], axis=1)
    df[f'Weekly % of total'] = round(df[f'Weekly {column} toll'] / df[f'Total {column}'] * 100, 3)
    # Plotly
    # 1. Table

    # TODO: Add table name and Remove margins: my-5
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
        page_size=9,
        style_header=
            {
            'fontWeight': 'bold',
            'border': 'thin lightgrey solid',
            'backgroundColor': 'rgb(100, 100, 100)',
            'color': 'white'
            },
        style_cell={
            'fontFamily': 'Helvetica',
            'textAlign': 'left',
            'width': '150px',
            'minWidth': '80px',
            'maxWidth': '100px',
            'whiteSpace': 'normal',#'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'backgroundColor': 'Rgb(230,230,230)'
            },
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in {'State'}
        ],
        
        # fixed_rows={'headers': True, 'data': 0}
    )

    # 2. Scatter plot
    weekly_plot = px.scatter(
        data_frame=df,
        x='Death to Case ratio',
        y='Total per capita (%)',
        size=f'Weekly % of total',
        color=f'Total {column}',
        title='Weekly increase by state',
        hover_name='State'
    )

    # 3. Bar chart
    bar_plot = px.bar(
        df,
        x='State',
        y=f'Total {column}',
        text='Total per capita (%)',
        title=f'Total {column} by states'
    )
    bar_plot.update_traces(texttemplate='%{text:0.2f}')
    bar_plot.update_layout(uniformtext_minsize=8, uniformtext_mode='show')

    return stats_table, weekly_plot, bar_plot

@app.callback(Output('container-button-timestamp', 'data'),
              [Input('btn-cases', 'n_clicks_timestamp'),
               Input('btn-deaths', 'n_clicks_timestamp')])
def displayClick(case_click, death_click):

    if int(death_click) > int(case_click):
        msg= 'deaths'
    else:
        msg = 'cases'
    return {'tab': msg} #html.Div(msg)

# 1. USA analysis
@app.callback(Output('cases_or_deaths_content', 'children'),
            [Input("container-button-timestamp", "data")]
)
def render_usa_data(data):
    column = data['tab']

    stats_table, weekly_plot, bar_plot = return_usa_stats(df, column)

    usa_plots = html.Div([

        html.Div(
        children=[
            html.Div(children=[
                stats_table
            ], className='col-md-6 my-2'
            ),
            html.Div(children=[
                    dcc.Graph(
                    figure=weekly_plot 
                    )
                ], className='col-md-6 px-0'
            )
        ], className='row mx-0'),

        html.Div(
        [
            html.Div(children=[
                    dcc.Graph(
                    id='usa-bar',
                    figure=bar_plot   
                    )
                ], className='col-md-12 px-0'
            ),
        ], className='row mx-0'),
       
    ]
    )
    return usa_plots
 
# 2. USA map - selected state
@app.callback(
    Output('states_content', 'figure'),
    [Input("container-button-timestamp", "data")]
)
def render_usa_map(data):
    column = data['tab']  # column: cases/ deaths

    state_df = df.groupby(['state']).max()
    state_df.reset_index(inplace=True)

    fig_usa_map = px.choropleth(
        state_df,
        locations='Postal',
        locationmode="USA-states",
        scope='usa',
        color=column,
        color_continuous_scale=color_scale[column],
        hover_name='state'   #'Postal'
        # ,hover_data=['state']
    )

    fig_usa_map.update_layout(
        title_text = f'USA {column}',
        clickmode='event+select'
    )

    return fig_usa_map

@app.callback(
    Output('clicked-state', 'data'),
    [Input('states_content', 'clickData')]
)
def store_clicked_state(clickData):
    return json.dumps(clickData, indent=2)


# 3. State-wise analysis and Time-Series
@app.callback(
    [
        Output('state_plot', 'figure'),
        Output('arima_plot', 'figure')
    ],
    [
        Input("container-button-timestamp", "data"),
        Input('clicked-state', 'data')
    ]
)
def statewise_plots(data, clickedState):
    '''
    data (dict) :: {'tab': str} can be 'cases' or 'deaths'
    clickedState (str) :: Click event is a string and first needs to be converted to dict
    {"points": [{"curveNumber": int, "pointNumber": int, "pointIndex": int,
                 "location": str, "z": int, "hovertext": str}]
    }
    location is 2-letter state code and hovertext is set to state name
    '''

    column = data['tab']
    full_df = df.copy()
    if clickedState == 'null':
        daily_df = full_df.groupby('date').sum()
        state = 'the United States'
    
    else:
        clicked_point = eval(clickedState)['points'][0]
        state = clicked_point['hovertext']
        daily_df = full_df[full_df['state']==state].set_index('date')

    # 1. State-wise plot
    fig_usa_timeline = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_usa_timeline.add_trace(
        go.Scatter(
        x = daily_df.index,
        y = daily_df[column].diff(),
        name=f"Daily {column}",
        fill='tozeroy',
        
        ), secondary_y = True
    )
    
    fig_usa_timeline.add_trace(
        go.Scatter(
        x = daily_df.index,
        y = daily_df[column],
        name=f"Total {column}",
        
        ), secondary_y = False
    )

    fig_usa_timeline.update_layout(
        title = f'Tally of {column} in {state}'
    )

    fig_usa_timeline.update_xaxes(
        rangeslider_visible=True,
    )

    # 2. TS Forecasting
    MA_period = 5
    
    daily_df[['daily_cases', 'daily_deaths']] = daily_df[['cases', 'deaths']].diff()

    # ARIMA predictions
    period = 30
    fig_arima_plot = TS_plot(daily_df.iloc[1:], column, model='arima')
    # 
    # LSTM predictions
    # lstm_result = TS_plot(df[[column, 'per_day', 'rolling_avg']], model='lstm')
    return fig_usa_timeline, fig_arima_plot

def TS_plot(df, column, model):
    '''
    column (str): deaths or cases
    '''

    time_steps = 7

    predictions=[]

    column_name = f'daily_{column}'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column_name],
        # name="Name of Trace 1"       # this sets its legend entry
    ))

    next_weekdays = pd.date_range(start=df.index[-1], periods=time_steps+1, freq='D')  # 8 days and start from the last day in data

    last_data = df[column_name].iloc[-1]   # Taking last value to continue the last point in the plot

    if model=='arima':
        title = 'Daily prediction with ARIMA model'
        
        model = pm.auto_arima(df[column_name], start_p=1, start_q=1,# d=d,
                     max_p=5, max_q=5, 
                     seasonal=False,
                     stepwise=True, suppress_warnings=True,# D=10, max_D=10,
                     error_action='ignore')

        # Create predictions for the future, evaluate on test
        preds, conf_int = model.predict(n_periods=time_steps, return_conf_int=True)

        # https://community.plotly.com/t/fill-area-upper-to-lower-bound-in-continuous-error-bars/19168
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], preds),
        mode='lines',
        name='Predicted'
        ))

        # Lower bound        
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], conf_int[:,0]),
        mode='lines',
        line=dict(width=0.5, color="rgb(141, 196, 26)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty'
        ))

        # Upper bound
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], conf_int[:,1]),
        mode='lines',
        line=dict(width=0.5,
                 color="rgb(255, 188, 0)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty'
        ))

    if model=='rnn':
        title = 'Daily prediction with LSTM newtwork'
        
        
        # Create predictions for the future, evaluate on test
        # preds = forecast(n_periods=time_steps, return_conf_int=True)


        pass

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        title = title
    )

    return fig

# Main Layout
app.layout = html.Div(children=[

    # Top Navbar
    dbc.Navbar(
        [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                # [dbc.Col(html.H1('Covid in the USA', className="mb-2", style={'text-align': 'center'}))],
                [dbc.Col(dbc.NavbarBrand("COVID-19 DASH", className="mb-2"))],
                align="center",
                # no_gutters=True,
            )
        )        
        ], color='dark', dark=True
    ),

    # USA all states 
    html.Div(children=[        
        html.Br(),
        dcc.Markdown(children=intro_markdown),
        html.Div(children=[        
            html.Button('Cases', id='btn-cases', n_clicks_timestamp=0, className="btn btn-secondary "),
            html.Button('Deaths', id='btn-deaths', n_clicks_timestamp=0, className="btn btn-secondary "),
            ],
            className='btn-group btn-group-toggle my-2', 
            style={'width': '10%','padding-left':'1%'}
        ),
        dcc.Store(id='container-button-timestamp')
        ]
    ),
    html.Div(id='cases_or_death_tabs'),
    html.H1('Analysis of the United States', className="mb-2", style={'text-align': 'center'}),                
    
    # TODO: https://community.plotly.com/t/data-table-select-all-rows/16619 -> Refer for un/selecting all rows
    html.Div(id='cases_or_deaths_content'),
    html.Br(),

    # USA map and statewise Time-series analysis
    html.H1('State-wise analysis', className="mb-2", style={'text-align': 'center'}),        

    dcc.Store(id='clicked-state'),

    html.Div([
        html.Div(
        [
            html.Div(children=[
                    dcc.Graph(
                    id='states_content'    
                    )
                ], className='col-md-6 px-0'
            ),
            html.Div(children=[
                    dcc.Graph(
                    id='state_plot',
                    )
                ], className='col-md-6 px-0'
            )
        ], className='row mx-0'),

        html.Div(
        [
            html.Div(children=[
                    dcc.Graph(
                    id='arima_plot'    
                    )
                ], className='col-md-6 px-0'
            ),
            # html.Div(children=[
            #         dcc.Graph(
            #         id='state_plot',
            #         )
            #     ], className='col-md-6 px-0'
            # )
        ], className='row mx-0'),  
    ])


    ], 
    style={'background-image': 'url("/assets/coronavirusbg.png")', 'background-size': '1700px 350px',
    # 'background-color': 'rgba(155, 5, 5, 0.16)'
    }
)

   

if __name__ == "__main__":
    app.run_server(debug=True)