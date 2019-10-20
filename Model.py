#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Model
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, MaxPool2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd


# In[18]:


# load csv file
data = pd.read_csv('train_foo.csv')
dataset = np.array(data)
np.random.shuffle(dataset)
X = dataset
Y = dataset
X = X[:,1:2501]
Y = Y[:, 0]


# In[19]:


# data scaling
X_train = X[0:12000, :]
X_train = X_train / 255.
X_test = X[12000:13201, :]
X_test = X_test / 255.


# In[10]:


Y =Y.reshape(Y.shape[0], 1)
Y_train = Y[0:12000, :]
Y_train = Y_train.T
Y_test = Y[12000:13201, :]
Y_test = Y_test.T


# In[12]:


print("Number of training examples: "+str(X_train.shape[0]))
print("Number of test examples: "+str(X_test.shape[0]))
print("X_train shape: "+str(X_train.shape))
print("Y_train shape: "+str(Y_train.shape))
print("X_test shape: "+str(X_test.shape))
print("Y_test shape: "+str(Y_test.shape))


# In[59]:


# one hot encoding
image_x = 50
image_y = 50

train_y = np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical(Y_test)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
print("X_train shape: "+str(X_train.shape))
print("X_test shape: "+str(X_test.shape))
print("train_y shape: "+str(train_y.shape))
print("test_y shape: "+str(test_y.shape))


# In[60]:


def keras_model(image_x, image_y):
    noc = 12
    cnn = Sequential()
    im_shape=(image_x, image_y,1)
    kernelSize = (3, 3)
    ip_activation = 'relu'
    ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
    cnn.add(ip_conv_0)
    ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1)
    ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1_1)

    pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    cnn.add(pool_1)
    
    drop_layer_0 = Dropout(0.2)
    cnn.add(drop_layer_0)
    
    flat_layer_0 = Flatten()
    cnn.add(Flatten())
    
    # Now add the Dense layers
    h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_0)
    # Let's add one more before proceeding to the output layer
    h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_1)
    op_activation = 'softmax'
    output_layer = Dense(units=noc, activation=op_activation, kernel_initializer='uniform')
    cnn.add(output_layer)
    opt='adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    # Compile the classifier using the configuration we want
    cnn.compile(optimizer=opt, loss=loss, metrics=metrics)
    return cnn
    
    '''
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(5,5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(noc, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath='ges_model.h5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list
    '''


# In[ ]:


model = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=6, batch_size=64)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%"%(100-scores[1]*100))
print_summary(model)

model.save('ges_model.h5')


# In[ ]:




