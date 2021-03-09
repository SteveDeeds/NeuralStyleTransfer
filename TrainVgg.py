import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.models import load_model

batch_size = 64
img_height = 64
img_width = 64

# batch_size = 64
# img_height = 512
# img_width = 512

#trainPath = "lessPaintings"
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

# Last block of layers
block5 = [
          'block5_conv1',
          'block5_conv2', 
          'block5_conv3', 
          'block5_conv4'
          ]

a_couple_style_layers = [
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
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical',
    subset='validation'
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "PaintingClassifier.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False)


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
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.Dense(256)(x)
predictions = Dense(len(classes), activation='softmax')(x)

# this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
model = Model(inputs=i, outputs=predictions)

#model = load_model("PaintingClassifier.hdf5")

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

model.fit(
    x=train_generator,
    steps_per_epoch=10,
    #batch_size=batch_size,
    validation_data=validation_generator,
    validation_steps=1,
    epochs=10,
    callbacks=[checkpoint])

# load the model because the last one might not be the best.
model = load_model("PaintingClassifier.hdf5")


model.trainable = True

# for layer in model.get_layer("vgg19").layers:
#     if layer.name in style_layers:
#         layer.trainable = True
#     else:
#         layer.trainable = False

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )
model.fit(
    x=train_generator,
    steps_per_epoch=10,
    #batch_size=batch_size,
    validation_data=validation_generator,
    validation_steps=10,
    epochs=10,
    callbacks=[checkpoint])

# load the model because the last one might not be the best.
model = load_model("PaintingClassifier.hdf5")
model.get_layer("vgg19").save("PaintingVgg19.hdf5")
