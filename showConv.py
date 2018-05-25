import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import glob
import sklearn
import cv2
import os
import scipy
import matplotlib.pyplot as plt
import sklearn.metrics

grandpa = cv2.imread("simpsons_dataset/abraham_grampa_simpson/pic_0000.jpg")
plt.figure()
plt.imshow(grandpa)

height, width, channels = grandpa.shape
print (height, width, channels)
grandpa = cv2.resize(grandpa,(64,64))

ann = Sequential()
x = Conv2D(filters=64,kernel_size=(3,3), padding='same',input_shape=(64,64,3))
ann.add(x)
ann.add(Activation('relu'))
##ann.add(Conv2D(64, (3, 3)))
##ann.add(Activation('relu'))
ann.add(MaxPooling2D(pool_size=(2, 2)))

ann.load_weights('test.h5')
"""x2 = Conv2D(filters=64,kernel_size=(3,3), padding='same')
ann.add(x2)
ann.add(Activation('relu'))
ann.add(MaxPooling2D(pool_size=(2, 2)))
ann.load_weights('test2.h5')"""



def nice_printer(model,simpson):
    print("origin size:")
    print(simpson.shape)
    conv_simpson = model.predict(np.expand_dims(simpson,axis=0))
    conv_simpson = np.squeeze(conv_simpson,axis=0)
    ##conv_simpson = conv_simpson.reshape(conv_simpson.shape[:2])
    print("after pooling size:")
    x2w = x.get_weights()[0][:,:,0:3,:]
    print(conv_simpson.shape)
    n = 1
    for i in range(1,6):
        for j in range(0,3):
            plt.subplot(5,4,n)
            plt.imshow(x2w[j][:,:,i],interpolation="nearest",cmap="gray")
            n += 1
        plt.subplot(5,4,n)
        plt.imshow(conv_simpson[:,:,i])
        n += 1

print(x.get_weights()[0].shape)
##print(x2.get_weights()[0].shape)


##first layer
x1w = x.get_weights()[0][:,:,0,:]
print(x1w.shape)
plt.figure()

for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")

#second layer



    #plt.subplot(8,8,i+1)



plt.figure()

nice_printer(ann,grandpa)

plt.show()

##ann.fit(X_train, y_train, epochs=5, batch_size=32)
