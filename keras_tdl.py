from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

#ONE TO ONE LSTM SEQUENCE PREDICTION
#prepare sequence
# length = 5
# seq = array([i/float(length) for i in range(length)])

# x = seq.reshape(len(seq),1, 5, 1)   #1 sample, 5 time steps, 1 feature
# y = seq.reshape(len(seq), 1, 5)

#define LSTM
# n_neurons = length
# n_batch = length
# n_epoch = 1000

#create LSTM
# model = Sequential() 
# model. add(LSTM(n_neurons, input_shape = (1,1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer = 'adam')
# print(model.summary())

#train LSTM
# model.fit(x,y, epochs = n_epoch, batch_size= n_batch, verbose=2)

#evaluate LSTM
# result = model.predict(x, batch_size=n_batch, verbose=0)
# for value in result:
#     print('%.1f' % value)


#MULTIPLE TO ONE SEQUENCE PREDICTION
# length = 5
# seq = array([i/float(length) for i in range(length)])

# x = seq.reshape(1, len(seq), 1)   #1 sample, 5 time steps, 1 feature
# y = seq.reshape(1, len(seq))

# n_neurons = length
# n_batch = 1
# n_epoch = 500

# model = Sequential() 
# model. add(LSTM(n_neurons, input_shape = (length,1)))
# model.add(Dense(length))
# model.compile(loss='mean_squared_error', optimizer = 'adam')
# print(model.summary())

# model.fit(x,y, epochs = n_epoch, batch_size= 1, verbose=2)

# result = model.predict(x, batch_size=n_batch, verbose=0)
# for value in result[0,:]:
#     print('%.1f' % value)


#MANY TO MANY LSTM WITH TIME DISTRIBUTED LAYER
#input myst be at least 3D
#output will be 3D
length = 5
seq = array([i/float(length) for i in range(length)])

x = seq.reshape(1, length, 1)   #1 sample, 5 time steps, 1 feature
y = seq.reshape(1, length, 1)

n_neurons = length
n_batch = 1
n_epoch = 1000

model = Sequential() 
model. add(LSTM(n_neurons, input_shape = (length,1), return_sequences=True))    #return a sequence of 5 outputs for each LSTM unit
model.add(TimeDistributed(Dense(1))) #single output value
model.compile(loss='mean_squared_error', optimizer = 'adam')
print(model.summary())

model.fit(x,y, epochs = n_epoch, batch_size= 1, verbose=2)

result = model.predict(x, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
    print('%.1f' % value)