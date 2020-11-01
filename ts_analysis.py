import numpy as np
import torch
from torch import nn 

class LSTM(nn.Module):

    def __init__(self, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 2
        self.n_cols = 1  

        self.lstm = nn.LSTM(input_size=self.n_cols, 
                            hidden_size=self.hidden_layer_size,
                            num_layers = self.num_layers, 
                            batch_first=True)  

        hidden_lin = 20

        self.linear1 = nn.Linear(self.hidden_layer_size, hidden_lin)
        self.drop = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(hidden_lin, self.n_cols)

    def forward(self, input_seq, start_bool=False):
       
        x = input_seq.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out.view(len(input_seq), -1))
        x = self.drop(x)
        x = self.linear2(x)

        return x[-1:]  # To get last prediction

device = torch.device('cpu')
model_cases = LSTM()
PATH = 'assets/lstm_cases.pt'
model_cases.load_state_dict(torch.load(PATH, map_location=device))

model_deaths = LSTM()
PATH = 'assets/lstm_deaths.pt'
model_deaths.load_state_dict(torch.load(PATH, map_location=device))

models = {'model_cases': model_cases, 'model_deaths': model_deaths}
print("Models loaded")

input_len=7

def preprocess(x,y):
  div = max(max(x), 1)
  x = x/div 
  y = y/div
  x = torch.tensor(x).view(-1,1).float().to(device)
  y = torch.tensor(y).view(1,1).float().to(device)
  return x,y,div

seq = np.array([0,0,1,1,2,3,2,1,4,5,6,9,7,10,15,12,20,17,30,31,35,40,38,45,50,66,55,
                70, 80,82,100,95,130,122,150,140,175,150,190,201,196,230,220,250,240,
                230,225,233,240,215,200,209,195,198,190,220,230,228,255,280,290,
                315,303,345,367,380,370,410,435,477,452,497,533,520,569,600])


def predict(seq, model, ind=30):
    j = ind + input_len
    x = seq[j-input_len:j]
    y = seq[j]
    print('X:', x)

    x,y,div = preprocess(x,y)

    model.eval()

    with torch.no_grad():
        ypred = model(x)

    y = y.squeeze()*div
    ypred = ypred.squeeze()*div
    print(y,ypred)  

predict(seq, models['model_deaths'])    
predict(seq, models['model_cases'])    