#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:41:02 2023

@author: libo
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn


__all__ = ["MyMode", "get_test_data"]

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

def get_test_data():
    file = '10persons.xlsx'
    df = pd.read_excel(file)

    df = df.fillna(0)
    df = df.iloc[:, 1:46]
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    X = np.asarray(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    return X_test[1].reshape(1,44)