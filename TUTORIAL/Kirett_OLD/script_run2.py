##Run this code to predict is a person is having a respiratory problem or not.
##A score below 0.5 show unhealth while a score of above 0.5 shows healthy...

from tvm.driver import tvmc
import numpy as np
import pandas as pd
import time

model = tvmc.load('/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/ANN-model.h5')

##An Healthy Person
#file = '/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/onepersongood.xlsx'

##An Unhealthy Person
file = '/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/oneperson.xlsx'

df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)
input_1 = X

###
start1 = time.time()
###

##Use this line to run with TUNE tune for the first time
#tvmc.tune(model, target="llvm", tuning_records="model.log")

##Use this line after the first time tune
#package = tvmc.compile(model, target="llvm", tuning_records="model.log")

##Use this line to run WITHOUT tune
package = tvmc.compile(model, target="llvm")

##Check result
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})

##
end1 = time.time()
print('Prediction Time: ',end1 - start1, 's')

#print('result: \n', result)
print(result.outputs)
