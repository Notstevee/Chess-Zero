
import numpy as np
import chess
import tensorflow as tf
from keras import backend




"""T=8
depth=14*T+7
def network(width=8,height=8,depth=14*T+7):
    input=tf.keras.Input(shape=(width,height,depth,))

model=keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(256,3,padding='same'))

print(model.summary())"""


def ResidualBlock(x:tf.Tensor):
    y=tf.keras.layers.Conv2D(256,3,padding='same')(x)
    y=tf.keras.layers.BatchNormalization()(y)
    y=tf.keras.layers.Activation(tf.keras.activations.relu)(y)
    y=tf.keras.layers.Conv2D(256,3,padding='same')(x)
    y=tf.keras.layers.BatchNormalization()(y)
    x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def PolicyHead(x:tf.Tensor):
    y=tf.keras.layers.Conv2D(256,3,padding='same')(x)
    y=tf.keras.layers.BatchNormalization()(y)
    y=tf.keras.layers.Activation(tf.keras.activations.relu)(y)
    y=tf.keras.layers.Conv2D(73,3,padding='same')(y)
    return y

def ValueHead(x:tf.Tensor):
    y=tf.keras.layers.Conv2D(1,1,padding='same')(x)
    y=tf.keras.layers.BatchNormalization()(y)
    y=tf.keras.layers.Activation(tf.keras.activations.relu)(y)

    y=tf.keras.layers.Dense(256, activation='relu')(y)
    y=tf.keras.layers.Activation(tf.keras.activations.relu)(y)
    y=tf.keras.layers.Flatten()(y)
    y=tf.keras.layers.Dense(1, activation='tanh')(y)
    return y
    

def NetTower():
    input=tf.keras.Input(shape=(8,8,119,))
    x=tf.keras.layers.Conv2D(256,3,padding='same')(input)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)


    for i in range(4):
        x=ResidualBlock(x)
    policy,value=PolicyHead(x),ValueHead(x)
    model=tf.keras.Model(inputs=input,outputs=[policy,value])
    
    return model

#print(NetTower(x))
#NetTower

#def CNN(input):
    

#CNN(1)
#https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188#