#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nrdenoise
import numpy as np
import pandas as pd
import os
import torch
import random
from matplotlib import pyplot as plt


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

load_data = torch.load('./model.pth', map_location=device)
model_name = load_data[ 'model_name' ]

if model_name == 'pCNN':
    model = nrdenoise.pCNN()
elif model_name == 'DnCNN':
    model = nrdenoise.DnCNN()
elif model_name == 'CAE':
    model = nrdenoise.CAE()

model.to(device)

if device == torch.device('cuda'):
    model = nn.DataParallel(model)

if device == torch.device('cuda'):
    model.module.load_state_dict( load_data['model_state_dict'] )
else:
    model.load_state_dict( load_data['model_state_dict'] )

scaler = load_data['scaler']

input_data_length = load_data['input_data_lenght']

x_df = pd.read_csv( './test_data.csv',    sep=',', engine='python', header=None )
y_df = pd.read_csv( './test_target.csv', sep=',', engine='python', header=None )

i = random.randint(0, len(x_df)-1)

x = x_df.iloc[i].values
y = y_df.iloc[i].values

if scaler != None:
    x = scaler.transform(x.reshape(1, -1)).reshape(-1)

x = torch.tensor(x, dtype=torch.float).unsqueeze(0).to(device).unsqueeze(0)

model.eval()

p = model(x)

x = x.cpu().data.numpy().reshape(-1)
p = p.cpu().data.numpy().reshape(-1)

if scaler != None:
    x = scaler.inverse_transform(x.reshape(1, -1)).reshape(-1)
    p = scaler.inverse_transform(p.reshape(1, -1)).reshape(-1)

np.set_printoptions(formatter={'float': '{:1.3e}'.format})

print('Input, Ground truth, Predicted')
for t in np.c_[x,y,p].tolist():
    print(', '.join([str(_) for _ in t]))

fig, ax = plt.subplots(figsize=(12, 4.5))
plt.rcParams["font.size"] = 12
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_xlabel('Channel')
ax.set_ylabel('Neutron count')
ax.plot(x, 'bo', label = 'Input')
ax.plot(y, 'g-', label = 'Ground truth')
ax.plot(p, 'ro', label = 'predicted')
ax.set_ylim([0.1, np.max(y)*10])
ax.set_yscale('log')
plt.legend(fontsize='small')
plt.plot()
plt.savefig(os.path.dirname(os.path.abspath(__file__))+'/out.png')
