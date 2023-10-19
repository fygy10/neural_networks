import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 50
img_height = 200
img_width = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(200, 200, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the weights of the pre-trained MobileNetV2 model
base_model.trainable = False

# Unfreeze specific layers for fine-tuning
for layer in base_model.layers[0:12]:
    layer.trainable = True

# Define the custom classification layers
transfer_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding="same", kernel_regularizer=l2(0.07)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding="same", kernel_regularizer=l2(0.07)),
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding="same", kernel_regularizer=l2(0.07)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.07)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.07)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Combine the base model with the custom classification layers
input_layer = tf.keras.layers.Input(shape=(200, 200, 3))
base_model_output = base_model(input_layer)
custom_model_output = transfer_model(base_model_output)
model = tf.keras.Model(inputs=input_layer, outputs=custom_model_output)

# compile model
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
# train model
EPOCH = 10
transfer_history = model.fit(train_ds, epochs=EPOCH,
                validation_data=val_ds,
                callbacks=[lr_schedule])

# visualize the training history and get the performance
plt.plot(transfer_history.history['accuracy'], label='train_accuracy')
plt.plot(transfer_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title("CNN Training")
plt.legend(loc='lower right')