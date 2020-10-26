import torch
import torch.nn as nn
import plotly.graph_objects as go
import pmdarima as pm
from pmdarima import model_selection

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

scaler = joblib.load('assets/scaler.gz')
MODEL_PATH = 'assets/model.pt'

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        print(self.hidden_layer_size)
        self.num_layers = 2
        self.n_cols = 1  # 2 columns of death and cases features being predicted

        self.lstm = nn.LSTM(input_size=self.n_cols, 
                            hidden_size=self.hidden_layer_size,
                            num_layers = self.num_layers, 
                            batch_first=True)  
        # No dataloader, but we unsqueeze 0th dimension for it

        hidden_lin = 20

        self.linear1 = nn.Linear(self.hidden_layer_size, hidden_lin)
        self.drop = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(hidden_lin, self.n_cols)

    def forward(self, input_seq, start_bool=False):
        # If not using dataloader, unsqueeze
        # if start_bool:
        #     self.initiate_hidden()

        x = input_seq.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out.view(len(input_seq), -1))
        x = self.drop(x)
        x = self.linear2(x)


        return x[-1:]  # To get last prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn_model = LSTM()
rnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

INPUT_WINDOW = 7

features = ['daily_cases', 'daily_deaths']

# Predicting LSTM
def predict_rnn(df, column, time_steps, arima_preds, INIT_WINDOW=0):

    data = df[features].iloc[-(INPUT_WINDOW+INIT_WINDOW):]

    preds = np.reshape(np.append(arima_preds, np.zeros_like(arima_preds)), (time_steps, -1), 'F')
    data = np.append(data, preds, axis=0)

    data = scaler.transform(data)
    data = torch.tensor(data)

    predictions = []

    rnn_model.eval()
    rnn_model.initiate_hidden()
    
    with torch.no_grad():
        for d in range(time_steps + INIT_WINDOW):
            # ypred = rnn_model(data.float())
            # data = torch.cat((data[1:], ypred), dim=0)
            seq = data[d:d+INPUT_WINDOW].float()
            if d >= INIT_WINDOW:
                print(seq)
            ypred = rnn_model(data.float())

            if d >= INIT_WINDOW:
                print(ypred)
                actual = scaler.inverse_transform(ypred)
                predictions.append(actual[0])

    predictions = np.ceil(np.array(predictions))
    
    return predictions

def get_predictions(df, column, time_steps):

    # 1. Daily

    # ARIMA
    # column_name = f'daily_{column}'

    arima_model = pm.auto_arima(
        df[column], start_p=1, start_q=1,# d=d,
        max_p=5, max_q=5, 
        seasonal=False,
        stepwise=True, suppress_warnings=True,# D=10, max_D=10,
        error_action='ignore'
    )

    # Create predictions for the future, evaluate on test
    arima_preds, conf_int = arima_model.predict(n_periods=time_steps, return_conf_int=True)

    predictions = predict_rnn(df, column, time_steps, arima_preds)    

    rnn_preds = predictions[:,0] if column == 'cases' else predictions[:,1]

    print(rnn_preds, arima_preds)

    
# Plot the required model
def TS_plot(df, model, **kwargs):
    '''
    column (str): deaths or cases
    '''

    column = kwargs.get('column')
    time_steps = 3

    predictions=[]

    daily_column = f'daily_{column}'

    get_predictions(df, daily_column, time_steps)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        # y=df[column_name],
        y=df[column],
        name="Latest"       # this sets its legend entry
    ))

    next_weekdays = pd.date_range(start=df.index[-1], periods=time_steps+1, freq='D')  # 8 days and start from the last day in data

    last_data = df[daily_column].iloc[-1]   # Taking last value to continue the last point in the plot

    if model=='arima_':
        title = 'Daily prediction with ARIMA model'
        
        # model = pm.auto_arima(df[column_name], start_p=1, start_q=1,# d=d,
        #              max_p=5, max_q=5, 
        #              seasonal=False,
        #              stepwise=True, suppress_warnings=True,# D=10, max_D=10,
        #              error_action='ignore')

        # # Create predictions for the future, evaluate on test
        # preds, conf_int = model.predict(n_periods=time_steps, return_conf_int=True)

        # https://community.plotly.com/t/fill-area-upper-to-lower-bound-in-continuous-error-bars/19168
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], preds),
        mode='lines',
        name='Forecast'
        ))

        # Lower bound        
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], conf_int[:,0]),
        name='Lower limit',
        mode='lines',
        line=dict(width=0.5, color="rgb(141, 196, 26)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty'
        ))

        # Upper bound
        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], conf_int[:,1]),
        name='Upper limit',
        mode='lines',
        line=dict(width=0.5,
                 color="rgb(255, 188, 0)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty'
        ))

    if model=='rnn_':
        title = 'Daily prediction with LSTM network'
        
        col_index = 0 if column=='cases' else 1
        
        # Create predictions for the future, evaluate on test
        preds = predict(df, time_steps)

        fig.add_trace(go.Scatter(
        x=next_weekdays,
        y=np.append([last_data], preds),
        mode='lines',
        name='Predicted'
        ))
        pass

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        title = title
    )

    return fig


if __name__ == "__main__":
        
    df = pd.read_csv('covid_state_9_27.csv')
    df = df[df.state=='Idaho']
    df[['daily_cases', 'daily_deaths']] = df[['cases', 'deaths']].diff()

    df['daily_cases'].fillna(df['cases'], inplace=True)
    df['daily_deaths'].fillna(df['deaths'], inplace=True)

    predictions = predict_rnn(df, 6, 0)
    print(predictions)

    col = 0
    actual = df[features].iloc[:,col]
    # print(actual)
    x = np.arange(len(actual), len(actual)+len(predictions))
    plt.plot(np.arange(len(actual)), actual)  
    plt.plot(x, predictions[:,col])  
    plt.show()