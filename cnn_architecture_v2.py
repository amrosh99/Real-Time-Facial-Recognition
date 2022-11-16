import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


################################### HOW TO RUN ###########################################################
'''
I ACHIEVED LOW LOSS AND HIGH ACCURACY (92%) WITH THE FOLLOWING PARAMETERS:

- target INPUT SHAPE = (64,64,3). Ours can be 1 for grayscale and 96x96. 
- num_classes is number of actors (11 in our case) and this is important for dense layer. Needs input = 11
- STEPS PER EPOCH = 128
- EPOCHS = 500
- OPTIMIZER: ADAM (LEARNING RATE =0.0001). Research shows Adam is the best and fastest optimizer for CNN
- Model.fit_generator used for fitting

'''

########################################################################################################

def CNN(input_shape, num_classes):
    # Build the network model
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=num_classes, activation='softmax')
    ])
    model.summary()
    learning_rate = 0.0001



    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



    #mymodel = model.fit_generator(train, steps_per_epoch=128, epochs=500)
    #mymodel

    #model.save("my_CNN_model.h5")
    #print("Saved model")

    #plot accuracy/loss
    #plt.figure()
    #plt.plot(mymodel.history['loss'])
    #plt.plot(mymodel.history['accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy/loss')
    #plt.xlabel('epoch')
    #plt.legend(['loss', 'accuracy'])

    return model













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
