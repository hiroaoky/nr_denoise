#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script is a sample code for the paper
    "Deep learning approach for interface structure analysis 
        with large statistical noise in neutron reflectometry"
    by H. Aoki, Y. Liu, T. Yamashita
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

########################################################################
class NRDDataset(Dataset):

    def __init__(self, x_df, y_df):
        self.x_df = x_df
        self.y_df = y_df

    def __getitem__(self, index):
        return (
            torch.tensor(self.x_df.iloc[index].values.astype(np.float32), dtype=torch.float).unsqueeze(0),
            torch.tensor(self.y_df.iloc[index].values.astype(np.float32), dtype=torch.float).unsqueeze(0)
        )

    def __len__(self):
        return self.x_df.shape[0]

########################################################################
class pCNN(nn.Module):

    def __init__(self):

        super(pCNN, self).__init__()

        self.input_channels  = [   1,  32,  64, 128, 256, 128,  64,  32 ]
        self.output_channels = [  32,  64, 128, 256, 128,  64,  32,   1 ]

        self.kernel_size = 5

        self.dilation    = 1

        self.stride      = 1

        self.padding     = 2

        self.bias        = True

        self.relu  = nn.ReLU(inplace=True)

        self.conv0a = nn.Conv1d(
            self.input_channels [0],
            self.output_channels[0],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn0 = nn.BatchNorm1d( self.output_channels[0] )

        self.conv1a = nn.Conv1d(
            self.input_channels [1],
            self.output_channels[1],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn1 = nn.BatchNorm1d( self.output_channels[1] )

        self.conv2a = nn.Conv1d(
            self.input_channels [2],
            self.output_channels[2],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn2 = nn.BatchNorm1d( self.output_channels[2] )

        self.conv3a = nn.Conv1d(
            self.input_channels [3],
            self.output_channels[3],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn3 = nn.BatchNorm1d( self.output_channels[3] )

        self.conv4a = nn.Conv1d(
            self.input_channels [4],
            self.output_channels[4],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn4 = nn.BatchNorm1d( self.output_channels[4] )

        self.conv5a = nn.Conv1d(
            self.input_channels [5],
            self.output_channels[5],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn5 = nn.BatchNorm1d( self.output_channels[5] )

        self.conv6a = nn.Conv1d(
            self.input_channels [6],
            self.output_channels[6],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn6 = nn.BatchNorm1d( self.output_channels[6] )

        self.conv7a = nn.Conv1d(
            self.input_channels [7],
            self.output_channels[7],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn7 = nn.BatchNorm1d( self.output_channels[7] )


    def forward(self, x):

        x_out = self.conv0a( x     )
        x_out = self.bn0   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv1a( x_out )
        x_out = self.bn1   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv2a( x_out )
        x_out = self.bn2   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv3a( x_out )
        x_out = self.bn3   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv4a( x_out )
        x_out = self.bn4   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv5a( x_out )
        x_out = self.bn5   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv6a( x_out )
        x_out = self.bn6   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv7a( x_out )
        x_out = self.bn7   ( x_out )
        x_out = self.relu  ( x_out )

        return x_out

########################################################################
class DnCNN(nn.Module):
    """
    DnCNN-like model (1D)
    """

    def __init__(self):

        super(DnCNN, self).__init__()

        self.input_channels  = [   1,  32,  64, 128, 256, 128,  64,  32 ]
        self.output_channels = [  32,  64, 128, 256, 128,  64,  32,   1 ]

        self.kernel_size = 5

        self.dilation    = 1

        self.stride      = 1

        self.padding     = 2

        self.bias        = True

        self.relu  = nn.PReLU()

        self.conv0a = nn.Conv1d(
            self.input_channels [0],
            self.output_channels[0],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn0 = nn.BatchNorm1d( self.output_channels[0] )

        self.conv1a = nn.Conv1d(
            self.input_channels [1],
            self.output_channels[1],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn1 = nn.BatchNorm1d( self.output_channels[1] )

        self.conv2a = nn.Conv1d(
            self.input_channels [2],
            self.output_channels[2],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn2 = nn.BatchNorm1d( self.output_channels[2] )

        self.conv3a = nn.Conv1d(
            self.input_channels [3],
            self.output_channels[3],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn3 = nn.BatchNorm1d( self.output_channels[3] )

        self.conv4a = nn.Conv1d(
            self.input_channels [4],
            self.output_channels[4],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn4 = nn.BatchNorm1d( self.output_channels[4] )

        self.conv5a = nn.Conv1d(
            self.input_channels [5],
            self.output_channels[5],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn5 = nn.BatchNorm1d( self.output_channels[5] )

        self.conv6a = nn.Conv1d(
            self.input_channels [6],
            self.output_channels[6],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )

        self.bn6 = nn.BatchNorm1d( self.output_channels[6] )

        self.conv7a = nn.Conv1d(
            self.input_channels [7],
            self.output_channels[7],
            kernel_size = self.kernel_size,
            dilation    = self.dilation,
            stride      = self.stride,
            padding     = self.padding,
            bias        = self.bias
        )


    def forward(self, x):

        x_out = self.conv0a( x     )
        x_out = self.bn0   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv1a( x_out )
        x_out = self.bn1   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv2a( x_out )
        x_out = self.bn2   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv3a( x_out )
        x_out = self.bn3   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv4a( x_out )
        x_out = self.bn4   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv5a( x_out )
        x_out = self.bn5   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv6a( x_out )
        x_out = self.bn6   ( x_out )
        x_out = self.relu  ( x_out )

        x_out = self.conv7a( x_out )

        return x + x_out

########################################################################
class Encoder(nn.Module):

    def __init__(
            self,
            input_channels   = [   1,  32,  64 ],
            output_channels  = [  32,  64, 128 ]
    ):

        super(Encoder, self).__init__()

        self.input_channels  = input_channels
        self.output_channels = output_channels

        self. conv1 = nn.Conv1d(
            self.input_channels [0],
            self.output_channels[0],
            kernel_size = 3,
            dilation    = 1,
            stride      = 1,
            padding     = 1,
            bias        = True
        )

        self.pool1 = nn.MaxPool1d(2)

        self.actf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            self.input_channels [1],
            self.output_channels[1],
            kernel_size = 3,
            dilation    = 1,
            stride      = 1,
            padding     = 1,
            bias        = True
        )

        self.pool2 = nn.MaxPool1d(2)

        self.actf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            self.input_channels [2],
            self.output_channels[2],
            kernel_size = 3,
            dilation    = 1,
            stride      = 1,
            padding     = 1,
            bias        = True
        )

        self.actf3 = nn.ReLU()

    def forward(self, x):

        x_block1 = self.conv1( x        )
        x_block1 = self.pool1( x_block1 )
        x_block1 = self.actf1( x_block1 )

        x_block2 = self.conv2( x_block1 )
        x_block2 = self.pool2( x_block2 )
        x_block2 = self.actf2( x_block2 )

        x_block3 = self.conv3( x_block2 )
        x_block3 = self.actf3( x_block3 )

        x_out = [x, x_block1, x_block2, x_block3]

        return x_out


class Decoder(nn.Module):

    def __init__(
            self,
            input_channels   = [ 128,  64,  32 ],
            output_channels  = [  64,  32,   1 ]
    ):

        super(Decoder, self).__init__()

        self.input_channels  = input_channels
        self.output_channels = output_channels

        self.up3to2 = UpProj1d(input_channels[0], output_channels[0])
        self.up2to1 = UpProj1d(input_channels[1], output_channels[1])
        self.up1to0 = UpProj1d(input_channels[2], output_channels[2])

    def forward(self, x):

        scale_size = [
            x[0].size(2),
            x[1].size(2),
            x[2].size(2),
            x[3].size(2),
        ]

        x_out = x[3]
        x_out = self.up3to2( x_out, scale_size[2] )
        x_out = self.up2to1( x_out, scale_size[1] )
        x_out = self.up1to0( x_out, scale_size[0] )

        return x_out


class UpProj1d(nn.Sequential):

    def __init__(self, n_in, n_out):

        super(UpProj1d, self).__init__()

        self.relu   = nn.ReLU(inplace=True)
        self.conv1a = nn.Conv1d(n_in,  n_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1a   = nn.BatchNorm1d(n_out)
        self.conv1b = nn.Conv1d(n_out, n_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b   = nn.BatchNorm1d(n_out)
        self.conv2  = nn.Conv1d(n_in,  n_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm1d(n_out)

    def forward(self, x, size):

        x = F.interpolate(x, size=size, mode='linear', align_corners=False)

        x_branch1 = self.relu( self.bn1a(self.conv1a(x)) )
        x_branch1 = self.bn1b( self.conv1b(x_branch1)    )
        x_branch2 = self.bn2 ( self.conv2(x)             )
        x_out     = self.relu( x_branch1 + x_branch2     )

        return x_out


class CAE(nn.Module):

    def __init__(self):

        super(CAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        x_encoded = self.encoder(x        )
        x_decoded = self.decoder(x_encoded)

        return x_decoded


