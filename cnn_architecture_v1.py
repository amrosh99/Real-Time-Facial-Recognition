#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amroshanshal
"""

import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop


'''

When using this layer as the first layer in a model, provide the keyword argument input_shape
(tuple of integers or None, does not include the sample axis), 
e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures

'''



def CNN(input_shape,num_classes):
   
    # Build the network model
    model = Sequential()
    
    # Conv2D is the numbers of filters that convolutional layers will learn from
    
    #conv 1
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    
    #The activition function helps in making the decision about which information should fire forward and which not by 
    #making decisions at the end of any network.
    
    #conv 2
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    #conv 3
    
    #Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. 
    #The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of 
    #better generalization and is less likely to overfit the training data.
    
    model.add(Conv2D(64, (1, 1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #Conv 4
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #conv 5
    model.add(Conv2D(64, (1, 1)))
    model.add(Activation("relu"))

      
    '''

      The output shape of the Dense layer will be affected by the number of neuron / units specified in the Dense layer. 
      For example, if the input shape is (8,) and number of unit is 16, then the output shape is (16,). 
      All layer will have batch size as the first dimension and so, input shape will be represented by (None, 8) and 
      the output shape as (None, 16). Currently, batch size is None as it is not set. Batch size is usually 
      set during training phase.

    '''

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    learning_rate = 0.001
    optimizer = RMSprop(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
   
              
    model.summary()
    return model

#this logic needs to be added somewhere for fitting the model
    
'''
x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.2, random_state = 0)

hist = model.fit(X_train, y_train,

          shuffle=True,

          batch_size=128,

          epochs=30,

          validation_data=(X_test, y_test)

         )
'''


# Then we have to input the images through an image data generator and fit the model on it


############ architecture explanation ##################
    

'''
Convolution neural network can broadly be classified into these steps:
    1. Input layer
    2. Convolutional layer 
    3. Subsampling
    4. Fully connected layers
    5. Output layers

Input layers are connected convolutional layers that perform tasks such as paddinng, striding,
the functioning of kernels. Essentially its the building block of the convolutional layer.

The convolutional layer is the responsible for extracting all the features from images and learn all 
the features an image has to offer. Thus, allowing for its use in object detection. The convolution layer computes the 
output of neurons that are connected to local regions or receptive fields in the input, each computing a dot product between 
their weights and a small receptive field to which they are connected to in the input volume. Each computation leads to extraction
of a feature map from the input image. 

The input layet will contain some pixel (in smaller scale like 3x3) values with some weight and height.
The kernels or filters will convolve around the input layer and give results which will retrieve all the 
features with fewer dimensions. 

The objective of subsampling is to get an input representation by reducing its dimensions, which helps in 
reducing overfitting. One of the techniques of subsampling is max pooling. Max pooling or average pooling reduces 
the parameters to increase the computation of our convolutional architecture.

By name, we can easily assume that max-pooling extracts the maximum value from the filter and average pooling takes 
out the average from the filter. We perform pooling to reduce dimensionality. We have to add padding only if necessary. 
we need to do padding in order to make our kernels fit the input matrices. Sometimes we do zero paddings, i.e. adding one row or
column to each side of zero matrices or we can cut out the part, which is not fitting in the input image, also known as valid padding.

The objective of the fully connected layer is to flatten the high-level features that are learned by convolutional layers and combining all the features. 
It passes the flattened output to the output layer where you use a softmax classifier or a sigmoid to predict the input class label.


How to determine the # of layers?
You cannot analytically calculate the number of layers or the number of nodes to use per layer in an artificial neural network to address a 
specific real-world predictive modeling problem. The number of layers and the number of nodes in each layer are model hyperparameters that you 
must specify and learn. You must discover the answer using a robust test harness and controlled experiments. 
Regardless of the heuristics, you might encounter, all answers will come back to the need for careful experimentation to see what works best for your specific
dataset.

https://www.analyticssteps.com/blogs/convolutional-neural-network-cnn-graphical-visualization-code-explanation
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
'''
########################################################
