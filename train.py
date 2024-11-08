#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nrdenoise

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
from tqdm import tqdm


model_name = 'DnCNN'
batch_size = 64
n_epoch    = 64
output_filename = 'model.pth'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

x_df = pd.read_csv( './training_data.csv',   sep=',', engine='python', header=None )
y_df = pd.read_csv( './training_target.csv', sep=',', engine='python', header=None )

if len(x_df.index) != len(x_df.index):
    print('ERROR:')
    print('len(x_df.index): {:}'.format(len(x_df.index)))
    print('len(y_df.index): {:}'.format(len(y_df.index)))
    exit()

scaler = MinMaxScaler((0, 1)).fit( pd.concat([x_df, y_df]))
x_df = pd.DataFrame(scaler.transform(x_df))
y_df = pd.DataFrame(scaler.transform(y_df))

i_train, i_valid, \
x_train, x_valid, \
y_train, y_valid  = train_test_split(
    np.arange(0, len(x_df.index)),
    x_df,
    y_df,
    test_size    = 0.2,
    random_state = None,
    shuffle      = True
)

print( 'Number of Training Data  :', len(x_train.index) )
print( 'Number of Validation Data:', len(x_valid.index) )

if model_name == 'pCNN':
    model = nrdenoise.pCNN()
elif model_name == 'DnCNN':
    model = nrdenoise.DnCNN()
elif model_name == 'CAE':
    model = nrdenoise.CAE()

model = model.to(device)

if device == torch.device('cuda'):
    model = nn.DataParallel(model)

model.verbose = False
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 8e-1 ** epoch)
loss_func1 = torch.nn.MSELoss()

dataset_train = nrdenoise.NRDDataset( x_train, y_train )
dataset_valid = nrdenoise.NRDDataset( x_valid, y_valid )

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

best_idx = 0
best_loss = float('inf')

loss_array_train = np.array([])
loss_array_valid = np.array([])

with tqdm(range(n_epoch), desc="Training (epoch)", leave=False, total=n_epoch) as progbar_global:

    for epoch_idx in progbar_global:

        model.train()
        loss_buffer = np.array([])
        with tqdm(dataloader_train, desc="Training (batch)", leave=False) as progbar_train:
            for batch in progbar_train:
                x, y = tuple(t.to(device) for t in batch)
                p = model(x)
                loss = loss_func1(p, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_buffer = np.append( loss_buffer, loss.cpu().detach() )
                progbar_train.set_postfix(i='"loss: {:1.2e}"'.format(loss_buffer.mean()))
            scheduler.step()
            loss_array_train = np.append( loss_array_train, loss_buffer.mean() )

        model.eval()
        loss_buffer = np.array([])
        with tqdm(dataloader_valid, desc="Validation      ", leave=False) as progbar_valid:
            for batch in progbar_valid:
                x, y = tuple(t.to(device) for t in batch)
                p = model(x)
                loss = loss_func1(p, y)
                loss_buffer = np.append( loss_buffer, loss.cpu().detach() )
                progbar_valid.set_postfix(i='"loss: {:1.2e}"'.format(loss_buffer.mean()))
            loss_array_valid = np.append( loss_array_valid, loss_buffer.mean() )

        if best_loss > loss_array_valid[-1]:
            best_loss = loss_array_valid[-1]
            best_idx = epoch_idx
            if device == torch.device('cuda'):
                state_dict = model.module.to('cpu').state_dict()
            else:
                state_dict = model.to('cpu').state_dict()
            model.to(device)
            save_data = {
                'model_name'       : model_name,
                'model_state_dict' : state_dict,
                'scaler'           : scaler,
                'input_data_lenght': len(x_df.index),
            }
            torch.save(save_data, output_filename)

        progbar_global.set_postfix(i='"loss: {:1.2e}"'.format(loss_array_valid[-1]))

loss = pd.DataFrame.from_dict(
        { 'epoch': np.arange(1, epoch_idx+2), 'train': loss_array_train, 'valid': loss_array_valid },
        orient='index'
).T

print('')
print('Loss:')
print('-'*50)
print('  epoch          train          valid')
for idx, row in loss.iterrows():
    print("   {:4}    {:1.5e}    {:1.5e}".format(row['epoch'], row['train'], row['valid']))
print('')

