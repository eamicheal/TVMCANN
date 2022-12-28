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
print('df_out:', df_out)
print('df_out:', df_out[0])
###

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)

###
start1 = time.time()
###
X_column = X.shape[0]
X_all = np.vsplit(X, X_column)

# the person you want to predict
Person_number = 3
Person_in_column = Person_number - 1
input_1 = X_all[Person_in_column]
###


###
errors = []
illness_results = []
for i in range(X_column):
    input_1 = X_all[i]
    # package = tvmc.compile(model, target="llvm", tuning_records="model.log")
    package = tvmc.compile(model, target="llvm")
    result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
    # Person_in_column = Person_in_column + 1
    # input_1 = X_all[Person_in_column]

    ###
    single_error = df_out[i] - result.outputs['output_0']
    errors.append(single_error)
    ###

    illness_results.append(result.outputs['output_0'])

end1 = time.time()
print('Prediction Time: ',end1 - start1, 's')
print('Result: \n', result)
print('illness_results: \n', illness_results)
###

###
print('errors: \n', np.sum(errors))
acc = 100*(1 - np.sum(errors)/X_column)
print('Accuracy: \n', acc, '%')

