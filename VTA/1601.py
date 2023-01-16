#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:33:20 2023

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

print(mod(torch.tensor( X_test[1]))) #orignal 
data = mod.x1.cpu().detach().numpy()
np.save("x1.npy", data)
data = mod.x2.cpu().detach().numpy()
np.save("x2.npy", data)
data = mod.x3.cpu().detach().numpy()
np.save("x3.npy", data)

mod.bypass_x1 = True
loadx1 = np.load("x1.npy")

print(mod(torch.tensor(loadx1))) #load and run with python


import tvm
from tvm import relay

mod = torch.load("MyMod.pth")

# hyperparameters
# input_shape = [1, 44]   #original input size
# my_input_data = X_test[1]

input_shape = [1, 24]   # L1 size
my_input_data = np.load("x1.npy")


mod.bypass_x1 = True
model = mod.eval()
# We grab the TorchScripted model via tracing

input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
    
from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(my_input_data.astype(dtype)))
# # Executeimg.
# m.run()
# # Get outputs
# # tvm_output = m.get_output(0)    
# tvm_output = m.get_output
# print(tvm_output)



num = 10  # number of times we run module for a single measurement
rep = 3  # number of measurements (we derive std dev from this)
timer = m.module.time_evaluator("run", dev, number=num, repeat=rep)
tcost = timer()

# # execute
m.run()

# get outputs
tvm_output = m.get_output(0)
print(tvm_output)

mean = tcost.mean * 1000
print("Average per sample inference time: %.2fms" % (mean))
print("max={:.2f}[ms], min={:.2f}[ms], median={:.2f}[ms], mean={:.2f}[ms], std={:.4f}[ms]".
      format(tcost.max*1000, tcost.min*1000, tcost.median*1000, tcost.mean*1000, tcost.std*1000))