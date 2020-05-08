'''
Created on 7. 5. 2020

@author: Jesusko
'''

if __name__ == '__main__':
    pass

import itertools
import logging
from datetime import datetime
import sys
import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler


np.set_printoptions(threshold=sys.maxsize)
dataframe  = pd.read_csv("FifaDataSetTestFloat+GK.csv", header=None ,sep=";" ,encoding='latin-1')
dataset = dataframe.values
TrainDataset = dataset[1001:17001,:]
TestDataset = dataset[:1001,:]
x_train = TrainDataset[:,2:36].astype(float) 
y_train = TrainDataset[:,36]
x_test = TestDataset[1:,2:36].astype(float)
y_test = TestDataset[1:,36]


"""
plt.scatter(X[0,100,Stplec_v_X],X[0:pocet_Prva,,Druhy_stplec_v_X]
plt.scatter(X[pocet_Prva:pocet_Druha,Stplec_v_X],X[pocet_Prva:poect_Druha,Druhy_stplec_v_X]
plt.scatter(X[pocet_Druha:pocet_Tretia,Stplec_v_X],X[poect_Druha:pocet_Tretia,Druhy_stplec_v_X]
plt.scatter(X[pocet_Tretia:XMAX,Stplec_v_X],X[pocet_Tretia:XMAX,Druhy_stplec_v_X]
"""  
df0 = dataframe[dataframe[36] == 0]
df1 = dataframe[dataframe[36] == 1]
df2 = dataframe[dataframe[36] == 2]
df3 = dataframe[dataframe[36] == 3]

df0=df0.values
df1=df1.values
df2=df2.values
df3=df3.values
print(df0[0:10])


plt.plot()
plt.scatter(df0[0:25,13], df0[0:25,14],color='blue',label='Brankar')
plt.scatter(df1[0:25,13], df1[0:25,14],color='green',label='Obranca')
plt.scatter(df2[0:25,13], df2[0:25,14],color='yellow',label='Zaloznik')
plt.scatter(df3[0:25,13], df3[0:25,14],color='red',label='Utocnik')
plt.title('')
plt.ylabel('GK handling')
plt.xlabel('GK diving')
plt.show()  

plt.plot()
plt.scatter(df0[0:25,11], df0[0:25,32],color='blue',label='Brankar')
plt.scatter(df1[0:25,11], df1[0:25,32],color='green',label='Obranca')
plt.scatter(df2[0:25,11], df2[0:25,32],color='yellow',label='Zaloznik')
plt.scatter(df3[0:25,11], df3[0:25,32],color='red',label='Utocnik')
plt.title('')
plt.xlabel('Finishing')
plt.ylabel('Standing tackle')
plt.show()  
     

"""
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""

logging.basicConfig(filename='log.log',
                            filemode='w',
                            #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            format='%(asctime)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

logging.info("Log Start")

y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 34-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=34))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.000005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

start = datetime.now()
#batch_size - po kolko prvkoch sa upravuju vahy
#epochs - konecne cislo epoch
history = model.fit(x_train, y_train,
          epochs=1,
          batch_size=100,
          verbose = 2,
          shuffle = True,
          use_multiprocessing=True
          )
end = datetime.now()
predict = model.predict_on_batch(x_test)
predict2 = model.predict(x_test, batch_size = 1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
errors = 0
for i in range(len(predict2)):
    p=predict2[i].copy()
    if p[0]>p[1] and p[0]>p[2] and p[0]>p[3]:
        p=[1,0,0,0]
    elif p[1]>p[0] and p[1]>p[2] and p[1]>p[3]:
        p=[0,1,0,0]
    elif p[2]>p[0] and p[2]>p[1] and p[2]>p[3]:
        p=[0,0,1,0]
    else:
        p=[0,0,0,1]
    
    comparison = p == y_test[i]
    equal_arrays = comparison.all()
    
    if not equal_arrays:
        errors=errors+1
        msg=(i,"finalna predikcia", p, "skutocnost", y_test[i] , "predikcia" , predict2[i] , "CHYBA")
        print(i,". finalna predikcia = ", p, "skutocnost = ", y_test[i] , " predikcia = " , predict2[i] , " CHYBA")
        logging.info(msg)
    else:
        msg=(i,"finalna predikcia", p, "skutocnost", y_test[i] , "predikcia" , predict2[i] , "USPECH")
        print(i,". finalna predikcia = ", p, "skutocnost = ", y_test[i] , " predikcia = " , predict2[i] , " USPECH")
        logging.info(msg)
 
msg=("pocet testovacich dat", len(y_test) , "pocet chyb", errors , "uspesnost", (1-(errors/len(y_test)))*100 , "%" , "cas trenovania" , str(end-start))
print("pocet testovacich dat =", len(y_test) , ", pocet chyb =", errors , ", uspesnost =", (1-(errors/len(y_test)))*100 , "%" , "cas trenovania " , end-start)
logging.info(msg)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()


