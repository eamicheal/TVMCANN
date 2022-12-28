from tvm.driver import tvmc
import numpy as np
import pandas as pd
import time

model = tvmc.load('/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/ANN-model.h5')
modelnew = tvmc.load('/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN-model2.h5')


###
#file = 'oneperson.xlsx'
#file = 'onepersongood.xlsx'
file = '/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/10persons.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)
###

###
X_column = X.shape[0]
X_all = np.vsplit(X, X_column)

# the person you want to predict
Person_number = 2
Person_in_column = Person_number - 1
input_1 = X_all[Person_in_column]

###
# tvmc.tune(model, target="llvm", tuning_records="model.log",)
# package = tvmc.compile(model, target="llvm", tuning_records="model.log")

###
start1 = time.time()
package = tvmc.compile(model, target="llvm")
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})

package = tvmc.compile(modelnew, target="llvm")
resultnew = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
end1 = time.time()

###
#print('reslut: \n', result)
print('reslut from model_new: \n', resultnew)
print('Prediction Time: ',end1 - start1, 's')

###
print('disease probability from model_old: \n', result.outputs['output_0'])
if result.outputs['output_0'] > 0.5 :
	print('Patient is sick of MODEL1')	
else:
	print('Patient is healthy of MODEL1')
print('disease probability from model_new: \n', resultnew.outputs['output_0'])

if resultnew.outputs['output_0'] > 0.5 :
	print('Patient is sick of MODEL2')	
else:
	print('Patient is health of MODEL2')


