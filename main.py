import numpy as np
from  tensorflow import keras
from os import listdir
import tensorflow as tf

def get_data_set(st):
    fin=open('datas/'+st)
    A=fin.readlines()
    A.pop(0)
    A.pop(len(A)-1)
    for i in range(0,len(A)):
        A[i]=A[i].replace('[','').replace('\n','').replace(' ','').replace(',','')
        A[i]=float(A[i])
    return A

model =tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd',loss="mean_squared_error")
dirs=listdir('E:\Gena\Python\PRED_NEURO\datas')

xs=np.array(get_data_set(dirs[0]),dtype=float)
ys=np.array([10,15,20],dtype=float)
model.fit(xs,ys,epochs=5)
results=model.predict([4,5,10])
print(results)


