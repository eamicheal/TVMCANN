from tvm.driver import tvmc
import numpy as np
import pandas as pd
from ProbVector import *

model = tvmc.load('/home/xilinx/tvm/vta/tutorials/codefiles/LCD/ANN-model.h5')

####
file = '/home/xilinx/tvm/vta/tutorials/codefiles/LCD/oneperson.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)
input_1 = X
###

#tvmc.tune(model, target="llvm", tuning_records="model.log")

#package = tvmc.compile(model, target="llvm", tuning_records="model.log")
package = tvmc.compile(model, target="llvm")
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
results = (result.get_output('output_0'))
print('reslut: \n', result)
DummyProbability(results)
