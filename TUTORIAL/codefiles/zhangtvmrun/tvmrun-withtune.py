from tvm.driver import tvmc
import numpy as np
import pandas as pd

model = tvmc.load('/home/xilinx/tvm/vta/tutorials/codefiles/zhangtvmrun/ANN-model.h5')


file = '/home/xilinx/tvm/vta/tutorials/codefiles/zhangtvmrun/onepersongood.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)
input_1 = X


#tvmc.tune(model, target="llvm", tuning_records="model.log")

package = tvmc.compile(model, target="llvm", tuning_records="model.log")
#package = tvmc.compile(model, target="llvm")
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
print('reslut: \n', result)
print(result.outputs)
