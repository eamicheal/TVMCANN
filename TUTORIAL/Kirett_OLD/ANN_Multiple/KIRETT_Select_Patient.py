from tvm.driver import tvmc
import numpy as np
import pandas as pd
import time

model = tvmc.load('/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Multiple/ANN-model.h5')

###
# file = 'oneperson.xlsx'
file = '/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Multiple/10persons.xlsx'
df = pd.read_excel(file)

###
df_out = df.iloc[:, 46]
#print('df_out:', df_out)
#print('df_out:', df_out[0])
###

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
Person_number = 1
Person_in_column = Person_number - 1
input_1 = X_all[Person_in_column]

###
#errors = []

#tvmc.tune(model, target="llvm", tuning_records="model.log",)
start1 = time.time()
package = tvmc.compile(model, target="llvm", tuning_records="model.log")
#package = tvmc.compile(model, target="llvm")
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
end1 = time.time()

    ###
#single_error = df_out[i] - result.outputs['output_0']
#errors.append(single_error)
    ###

print('Prediction Time: ',end1 - start1, 's')
print('Result: \n', result)
print(result.outputs['output_0'])

###
#print('errors: \n', np.sum(errors))
#acc = 100*(1 - np.sum(errors)/X_column)
#print('Accuracy: \n', acc, '%')
