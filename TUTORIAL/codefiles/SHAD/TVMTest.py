import tarfile

import tensorflow as tf
import keras
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import warnings
from numpy import loadtxt
from numpy import save,load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tvm
from tvm.driver import tvmc
from tvm import relay
import onnxruntime

# onnx_model_name = 'tvm_test.onnx'
#
# model = load_model('ANN-model.h5')
#
# onnx_model,_ = tf2onnx.convert.from_keras(model)
# onnx.save_model(onnx_model, onnx_model_name)

#Data Preparation

file = 'NHertz.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.savez("test.npz", input= X_test[0:1,:])
input_1=X_test[3:4,:]
save('d.npy',input_1)

#converts a machine learning model from a supported framework into TVM’s high level graph representation language called Relay.
model= tvmc.load('ANN-model.h5', shape_dict={'input':[44,]})
model.summary()

package= tvmc.compile(model, target= "llvm", output_format= 'so')
#package= tvmc.compile(model, target= "llvm", output_format= 'so', mod_name='mtom', package_path='/home/Shad/Desktop/TVM/cmodel.so')
#result= tvmc.run(package, device= "cpu")
print('Package is compiled')
ip1= load('d.npy')
pre= tvmc.run(inputs= {'input_1':load('d.npy')}, device='cpu', tvmc_package=package)
#pre= tvmc.run(inputs= {'input_1':d.npy}, hostname= '192.168.2.99', port= 9091, rpc_key='pynq', device='vta', tvmc_package=package)
print('TVM RUN is going on..')
print('TVM prediction:', pre.get_output('output_0'))


# NModel= load_model('ANN-model.h5')
# score= NModel.predict(input_1)
# print('Original model prediction: ',score)
