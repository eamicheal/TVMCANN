###SCRIPT TO PREDICT ANN FROM KIRETT DATASET

from tvm.driver import tvmc
import numpy as np
import pandas as pd
import time

###SELECT MODEL AND LOAD INTO TVM
model = tvmc.load('/home/xilinx/tvm/vta/tutorials/Kirett_OLD/ANN_Single/ANN-model.h5')
#model = tvmc.load('/home/xilinx/tvm/vta/Kirett/ANN-model_new.h5')
###

### SPECIFYING THE FILE AND PROFILLING
file = '/home/xilinx/tvm/vta/Kirett/10persons.xlsx'
df = pd.read_excel(file)

df = df.fillna(0)
df = df.iloc[:, 1:46]
y = df.iloc[:, -1]
X = df.iloc[:, :-1]
X = np.asarray(X).astype(np.float64)
###

### SELECT PATIENTS FROM A LIST OF 10
X_column = X.shape[0]
X_all = np.vsplit(X, X_column)

# the person you want to predict, #2 has cardio, #5 has no cardio
Person_number = 2
Person_in_column = Person_number - 1
input_1 = X_all[Person_in_column]
###

###OPTIONAL: TUNE MODEL
# tvmc.tune(model, target="llvm", tuning_records="model.log",)
# package = tvmc.compile(model, target="llvm", tuning_records="model.log")
###

### COMPILE AND RUN MODEL
start1 = time.time()
package = tvmc.compile(model, target="llvm")
result = tvmc.run(package, device="cpu", inputs={'input_1':input_1})
end1 = time.time()
###

### 5-COMPLICATIONS PROBABILITY
other_sum = 1 - result.outputs['output_0']
weight_array = np.random.dirichlet(np.ones(4))

respiy = other_sum * weight_array[0]
psychi = other_sum * weight_array[1]
abdomi = other_sum * weight_array[2]
others = other_sum * weight_array[3]

print('Cardio: ', result.outputs['output_0'])
print('Respiy: ', respiy)
print('Psychi: ', psychi)
print('Abdomi: ', abdomi)
print('Others: ', others)

if result.outputs['output_0'] > 0.5 :
	print('Patient suspected with CARDIOVASCULAR Disease')	
else:
	print('Patient has no CARDIOVASCULAR Disease')
	
if respiy > 0.5 :
	print('Patient suspected with RESPIRATORY Disease')	

if psychi > 0.5 :
	print('Patient suspected with PSYCHIATRIC Disease')
	
if abdomi > 0.5 :
	print('Patient suspected with ABDOMINAL Disease')
	
if others > 0.5 :
	print('Patient suspected with OTHER Diseases')

if ((result.outputs['output_0'] < 0.5) and (respiy < 0.5) and (psychi < 0.5) and (abdomi < 0.5) and (others < 0.5)):
        print('NOTHING FOUND, PATIENT SEEMS TO BE HEALTHY')
###

###TIME SUMMARY
print('-----------')
print(result)
print('Linux Run_time: ',end1 - start1, 's')
###
