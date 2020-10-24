import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

scaler = joblib.load('assets/scaler.gz')
MODEL_PATH = 'assets/model.pt'

class LSTM(nn.Module):
    def __init__(self, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 2
        self.n_cols = 2  # 2 columns of death and cases features being predicted

        self.lstm = nn.LSTM(input_size=self.n_cols, 
                            hidden_size=self.hidden_layer_size,
                            num_layers = self.num_layers, 
                            batch_first=True)  
        # No dataloader, but we unsqueeze 0th dimension for it

        self.linear = nn.Linear(self.hidden_layer_size, self.n_cols)

        self.initiate_hidden()

    def initiate_hidden(self):
        self.hidden_cell = (torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device),
                            torch.zeros(self.num_layers,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq, start_bool=False):
        # If not using dataloader, unsqueeze
        if start_bool:
            self.initiate_hidden()

        x = input_seq.unsqueeze(0)
        lstm_out, (h_0, c_0) = self.lstm(x, self.hidden_cell) # 

        h_0.detach_()
        c_0.detach_()
        self.hidden_cell = (h_0, c_0)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1:]  # To get last prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn_model = LSTM()
rnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

INPUT_WINDOW = 7
features = ['daily_cases', 'daily_deaths']

df = pd.read_csv('covid_state_9_27.csv')
df = df[df.state=='Florida']
df[['daily_cases', 'daily_deaths']] = df[['cases', 'deaths']].diff()

df['daily_cases'].fillna(df['cases'], inplace=True)
df['daily_deaths'].fillna(df['deaths'], inplace=True)


def predict(df, time_steps, INIT_WINDOW=10):

    data = df[features].iloc[-(INPUT_WINDOW+INIT_WINDOW):]

    data = scaler.transform(data)
    data = torch.tensor(data)

    predictions = []

    rnn_model.eval()
    rnn_model.initiate_hidden()
    
    with torch.no_grad():
        for d in range(time_steps + INIT_WINDOW):
            ypred = rnn_model(data.float())
            data = torch.cat((data[1:], ypred), dim=0)

            if d >= INIT_WINDOW:
                # print(ypred)
                actual = scaler.inverse_transform(ypred)
                predictions.append(actual[0])

    predictions = np.ceil(np.array(predictions))
    
    return predictions

predictions = predict(df, 6, 4)
print(predictions)

col = 0
actual = df[features].iloc[:,col]
# print(actual)
x = np.arange(len(actual), len(actual)+len(predictions))
plt.plot(np.arange(len(actual)), actual)  
plt.plot(x, predictions[:,col])  
plt.show()