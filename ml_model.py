import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib


scaler = joblib.load('assets/scaler.gz')
MODEL_PATH = 'assets/lstm.pt'

class LSTM(nn.Module):
    def __init__(self, n_cols=1, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.n_cols = n_cols

        self.lstm = nn.LSTM(input_size=n_cols, 
                            hidden_size=hidden_layer_size,
                            num_layers = self.num_layers, 
                            batch_first=True)  
        # No dataloader, but we unsqueeze 0th dimension for it

        self.linear = nn.Linear(hidden_layer_size, n_cols)

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
        return predictions[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_cols = 2
model = LSTM(n_cols)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

def predict(df):
    pass