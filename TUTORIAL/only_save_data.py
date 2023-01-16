#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:03:50 2022

@author: libo
"""

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn


file = '10persons.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


class MyMode(nn.Module):
    def __init__(self):
        super(MyMode,self).__init__() 
        self.LayerX1 = nn.Sequential(
            nn.Linear(in_features=44, out_features=24),
            nn.ReLU())
        self.LayerX2 = nn.Sequential(
            nn.Linear(in_features=24, out_features=10),
            nn.ReLU())
        self.LayerX3 = nn.Sequential(            
                # nn.Dropout(),
                nn.Linear(in_features=10, out_features=1),
                nn.Sigmoid(),
            )
        self.bypass_x1 = False
        self.bypass_x2 = False
        self.bypass_x3 = False
        
    def forward(self, x):
        if self.bypass_x1:
            self.x1 = x
        else:
            self.x1 = self.LayerX1(x)
            
        if self.bypass_x2:
            self.x2 = x
        else:
            self.x2 = self.LayerX2(self.x1)
            
        if self.bypass_x3:
            self.x3 = x
        else:
            self.x3 = self.LayerX3(self.x2)

        return self.x3


mod = torch.load("MyMod.pth")

data = mod.x1.cpu().detach().numpy()
np.save("x1.npy", data)
data = mod.x2.cpu().detach().numpy()
np.save("x2.npy", data)
data = mod.x3.cpu().detach().numpy()
np.save("x3.npy", data)
