import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cnn_architecture_v2 import CNN
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

filename = 'CNN_model_trained.pickle'

df = pd.DataFrame(pd.read_pickle('pca_faces.pickle'))

input_shape = (96, 96, 1)
num_classes = len(np.unique(df['names']))

faces = df['image_array']

# Mapping names to numbers
ids = df['names']
ids_dic = {v:k for k,v in enumerate(np.unique(ids))}
ids_mapped = ids.map(ids_dic)

# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(faces, ids_mapped, test_size = 0.2, random_state = 42)

# Convert X_train to a array
for i in range(len(X_train)):
    X_train.iloc[i] = np.asarray(X_train.iloc[i]).astype(np.float32)[:,:,np.newaxis]

# Convert X_test to a array
for i in range(len(X_test)):
    X_test.iloc[i] = np.asarray(X_test.iloc[i]).astype(np.float32)[:,:,np.newaxis]

X_train = np.array([np.array(val) for val in X_train])
X_test = np.array([np.array(val) for val in X_test])

# One hot encoding labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Using custom CNN
model = CNN(input_shape,num_classes)

# Fitting model
fitted = model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=500, validation_data=(X_test, y_test), use_multiprocessing=True)

# Saving fitted model
model.save("CNN_model.h5")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fitted.history['loss'])
plt.plot(fitted.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

