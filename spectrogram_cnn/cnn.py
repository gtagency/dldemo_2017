import keras
from keras.layers import Inputs
import numpy as np

def classification_cnn(input_shape, num_classes, num_filters, num_conv_layers):
    inputs = Inputs(input_shape)
    x = inputs
    for i in range(num_conv_layers):
        x = Conv2D(self.num_filters, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(num_classes)
    if num_classes > 1:
        x = Activation('softmax')(x)
    else:
        x = Activation('sigmoid')(x)
    return Model(inputs=inputs, outputs=x)
