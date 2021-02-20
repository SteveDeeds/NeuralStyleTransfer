import functools
import time
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, losses, models
from tensorflow.python.keras.preprocessing import image as kp_image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from tensorflow.keras.models import load_model

batch_size = 64
img_height = 512
img_width = 512

trainPath = "paintings"

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

classes = os.listdir(trainPath)
num_classes = len(classes)

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    horizontal_flip=True,
    rotation_range=10.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(img_height, img_width),
    batch_size=20,
    classes=classes,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(img_height, img_width),
    batch_size=20,
    classes=classes,
    class_mode='categorical',
    subset='validation'
)

# add an input layer
i = Input([img_height, img_width, 3])
x = tf.keras.applications.vgg19.preprocess_input(i)
# create the base pre-trained model
base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
base_model.summary()
base_model.trainable = False
x = base_model(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.Dense(256)(x)
predictions = Dense(len(classes), activation='softmax')(x)

# this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
model = Model(inputs=i, outputs=predictions)

#model = load_model("PaintingClassifier.hdf5")

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "PaintingClassifier.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False)

model.fit(
    x=train_generator,
    steps_per_epoch=20,
    validation_data=validation_generator,
    validation_steps=10,
    epochs=20,
    callbacks=[checkpoint])

model.save("PaintingClassifier.hdf5")

# set style layers to trainable
for name in style_layers:
  model.get_layer("vgg19").get_layer(name).trainable = True

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    x=train_generator,
    steps_per_epoch=20,
    validation_data=validation_generator,
    validation_steps=10,
    epochs=20,
    callbacks=[checkpoint])
  
model.get_layer("vgg19").save("PaintingVgg19.hdf5")
