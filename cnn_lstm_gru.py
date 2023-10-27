from tensorflow.keras import layers, models, utils, callbacks, optimizers, datasets

def generic_vns_function(rnn_type = '', num_classes=10):
    input_shape = (3, 28, 28, 1)  #specific for MNIST
    Input = layers.Input(shape=input_shape) #input layer
    
    #CNN; x updated at each step
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu'))(Input)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    #RNN; specific RNN layers to choose from
    if rnn_type == 'lstm':
        RNN_output = layers.LSTM(64)(x)
    elif rnn_type == 'gru':
        RNN_output = layers.GRU(64)(x)
    else:
        RNN_output = layers.SimpleRNN(64)(x) 

    #DNN 
    Output = layers.Dense(num_classes, activation='softmax')(RNN_output)
    
    model = models.Model(inputs=Input, outputs=Output)
    
    return model

def train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, 
              batch_size=batch_size, verbose=1, callbacks=cb)
    scores = model.evaluate(X_test, y_test, verbose=2)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    return model

def choose_dataset(dataset_type):
    if dataset_type == "speech_recognition":
        (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0   #normalize the data

        train_limit = (X_train.shape[0] // 3) * 3   #samples are divisible by 3 for the sequence length
        test_limit = (X_test.shape[0] // 3) * 3

        X_train = X_train[:train_limit].reshape(-1, 3, 28, 28, 1) #sequences in batch, sequence length, height, width, number of channels
        y_train = y_train[:train_limit].reshape(-1, 3)[:, 0]
        X_test = X_test[:test_limit].reshape(-1, 3, 28, 28, 1)
        y_test = y_test[:test_limit].reshape(-1, 3)[:, 0]
    else:
        raise ValueError("Invalid dataset type.")

    y_train, y_test = utils.to_categorical(y_train, num_classes=10), utils.to_categorical(y_test, num_classes=10)
    return (X_train, y_train), (X_test, y_test)

rnn_type = 'lstm'

def main():
    epochs = 10
    batch_size = 200
    lr = 0.0001
    dataset_type = "speech_recognition"
    
    (X_train, y_train), (X_test, y_test) = choose_dataset(dataset_type)
    
    opt = optimizers.Adam(lr=lr)
    model = generic_vns_function(rnn_type=rnn_type)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()