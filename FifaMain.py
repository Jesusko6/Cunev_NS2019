'''
Created on 5. 5. 2020

@author: Jesusko
'''

if __name__ == '__main__':
    pass

import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import Sequential
from keras import optimizers
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

#matplotlib inline

from keras.layers.advanced_activations import LeakyReLU

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(34, input_dim=34, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



dataframe  = pd.read_csv(r"C:\Users\Jesusko\Desktop\FifaDataSetTestFloat.csv", header=None ,sep=";")
dataset = dataframe.values
dataset = dataset[1:,:]
dataset = shuffle(dataset)
#print(training_data.tail(5))
x = dataset[:,2:36].astype(float)
y = dataset[:,36]
print(x)
print(y)

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=50, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True)

#results = cross_val_score(estimator, x, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

classifier = Sequential()
classifier.add(Dense(34, activation='relu', kernel_initializer='random_normal', input_dim=34))
classifier.add(Dense(120, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer ="adam",loss='mean_squared_error', metrics =['accuracy'])
history = classifier.fit(x_train,y_train, batch_size=1, epochs=50, validation_data=[x_test, y_test])

eval_model=classifier.evaluate(x_train, y_train)
print(eval_model)
print("\nAccuracy: %.2f%%\nLoss: %.2f%%" % (eval_model[1]*100, eval_model[0]*100))

y_pred = classifier.predict_classes(x_test)
print(y_pred)
print(y_test)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

