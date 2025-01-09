import os
import sys
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
from tensorflow import keras

class model_build():

    def build_model(self, num_classes=6):
        try:
 
            base_model = keras.applications.VGG16(
                weights='imagenet',
                input_shape=(224, 224, 3),
                include_top=False
                )

            base_model.trainable = False

            inputs = keras.Input(shape=(224, 224, 3))

            x = base_model(inputs, training=False)

            # Add pooling layer or flatten layer
            pooling = keras.layers.GlobalAveragePooling2D()(x)

            # Add final dense layer
            outputs = keras.layers.Dense(256, activation = 'softmax')(pooling)

            # Combine inputs and outputs to form the model
            model = keras.Model(inputs,outputs)

            model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
            
            logging.info("Model built successfully.")

            return model

        except Exception as e:
            raise CustomException(e, sys)