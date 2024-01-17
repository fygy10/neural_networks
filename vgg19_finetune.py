import tensorflow as tf
import tensorflow_datasets as tfds
import keras.layers as layers
import matplotlib.pyplot as plt 
from tensorboard import data as tf_data
import keras


train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True
)

# print(f"Number of training samples: {train_ds.cardinality()}")
# print(f"Number of validation samples: {validation_ds.cardinality()}")
# print(f"Number of test samples: {test_ds.cardinality()}")


# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(train_ds.take(9)):
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(image)
#     plt.title(int(label))
#     plt.axis("off")


#RESIZE ALL IMAGES TO 150 X 150
resize_fn = layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))


#RANDOM DATA AUGMENTATION FOR VARIANCE
#applied in the data_augmentation function
augmentation_layers = [
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


#SET BATCH SIZE AND PREFETCH DATA FOR FASTER PERFORMANCE
#stored in local memory (cache()) for faster performance
batch_size = 64
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()



#BUILD MODEL

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False,      #ensures only includes feature extraction layers not classification layers
)

#set to False when training for the first or deploying
base_model.trainable = True 

inputs = keras.Input(shape=(150, 150, 3))

#normnalize input to a range of -1 to 1 -> 127.5 allows this since it is between 1 and 255 and substracting -1 centers values around 0
scale_layer = layers.Rescaling(scale=1 / 127.5, offset= -1)
x = scale_layer(inputs)

#explicit forward base of model layers
x = base_model(x, training = False)     #keep as False to keep critical layers in inference mode so they do not update
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)


#COMPILE MODEL
#binary used since it is cats vs dogs
model.compile(
    optimizer = keras.optimizers.Adam(1e-5),
    loss = keras.losses.BinaryCrossentropy(from_logits=True),   #logits: raw outputs before activation function applied
    metrics = [keras.metrics.BinaryAccuracy()]
)

epochs = 1
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

model.evaluate(test_ds)