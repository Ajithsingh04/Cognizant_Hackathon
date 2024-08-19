from preprocessing import *
from model import *
import pandas as pd
import numpy as  np
import torch

from torch.nn.utils.rnn import pad_sequence

def collate_fn(xx):
  x_lens=[len(xx[i]) for i in range(len(xx))]
  xx_pad = pad_sequence(xx,batch_first=True,padding_value=0)

  return xx_pad,torch.tensor(x_lens,dtype=torch.long)

tdata = [
    
    [[1, 25, 60, 95, 36.1, 70, 12, 7, 98, 0.7, 75, 43, 13.7, 4500, 150000],
    [1, 25, 75, 97, 37.2, 87, 18, 16, 102, 1.1, 89, 48, 16.8, 7890, 275000] ],[[0, 20, 56, 95, 36.1, 70, 12, 7, 95, 0.7, 75, 43, 13.7, 4500, 150000]]
]

tensors = []
for i in range(len(tdata)):
    DataFrame = pd.DataFrame(tdata[i], columns=['Gender','Age','HR','O2Sat','Temp','MAP','Resp','BUN','Chloride','Creatinine','Glucose','Hct','Hgb','WBC','Platelets'])
    print(DataFrame)
    impute_logic(DataFrame)
    data_np = DataFrame.to_numpy()
    data_tensor = torch.from_numpy(data_np).long()
    tensors.append(data_tensor)

x_pad,x_lens = collate_fn(tensors)

print(x_pad.shape)




model = create_model()

model.eval()

with torch.no_grad():
    y = model(x_pad)


for i in range(len(x_pad)):
    predictions = F.softmax(y[:x_lens[i]], dim=-1)
    print(predictions,i)
    # Print probabilities
    probs = predictions.numpy()
    print('Predictions (Probabilities):')
    print(probs)