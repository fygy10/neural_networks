#BASIC NEURAL NETWORK

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the dataset
input_shape = (28 * 28,)    #define the input shape (specific for MNIST dataset)
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0    #normalize by converting to float32 and pixel values
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
y_train = to_categorical(y_train)   #one-hot encoding
y_test = to_categorical(y_test)

#how the model is fed forward
model = Sequential()

#model composition
model.add(Dense(units = 1000, activation = 'relu', input_shape = input_shape))
model.add(Dense(units=10, activation='softmax'))

#model compile
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

#training the model
model.fit(x_train, y_train, epochs=10, batch_size=200)

#evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

model.predict(x_test[:4])
y_test[:4]