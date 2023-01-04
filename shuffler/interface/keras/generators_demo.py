'''
A demo for running Keras inference on a database, and recording some results.
'''

import numpy as np
import cv2  # Used to resize objects to the same size.
import tensorflow as tf

from shuffler.utils import testing as testing_utils
from shuffler.interface.keras import generators


def make_model(input_shape, num_classes):
    ''' Make a simple two-layer convolutional model. '''
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.summary()
    return model


def main():
    # This database contains 3 images with 2 cars and 1 bus.
    in_db_file = testing_utils.Test_carsDb.CARS_DB_PATH
    rootdir = testing_utils.Test_carsDb.CARS_DB_ROOTDIR

    # Objects are resized to this shape.
    width = 100
    height = 100

    # The transform resizes every image, and makes the label categorical.
    transform_group = {
        'image': lambda x: cv2.resize(x, (width, height)),
        'name': lambda x: 1 if x == 'bus' else 0
    }

    # Make a generator of OBJECTS. Every returned item is an object in the db.
    generator = generators.ObjectGenerator(in_db_file,
                                           rootdir=rootdir,
                                           used_keys=['image', 'name'],
                                           transform_group=transform_group,
                                           batch_size=2,
                                           shuffle=False)

    model = make_model(input_shape=(height, width, 3), num_classes=2)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    epochs = 10
    model.fit(generator, epochs=epochs, workers=1)


if __name__ == '__main__':
    main()
