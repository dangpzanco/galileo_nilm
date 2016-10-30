from keras.models import Sequential, load_model
from keras.layers import Input, Activation, Flatten, Reshape, Dense, Conv1D, Dropout, Embedding
import keras.optimizers as opt
from keras.regularizers import l1, l2, activity_l1, activity_l2
from keras import backend as K
from keras.callbacks import Callback
from keras.utils.visualize_util import plot as keras_plot
from keras.metrics import fbeta_score


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

# get training data
hdf_filename = 'kettle1.h5'
df = pd.read_hdf(hdf_filename, key='active_power')

# check for nans
nan_check = np.isnan(np.sum(np.array(df['kettle'].values)))
if nan_check:
    exit()

# Dataset limits without gaps
# begin_train = pd.Timestamp('2013-2-14')
# end_train = pd.Timestamp('2015-8-18')

begin_train = pd.Timestamp('2013-8-27')
end_train = pd.Timestamp('2013-8-29')

mains_train = np.array(df['mains'][begin_train:end_train]).ravel()
kettle_train = np.array(df['kettle'][begin_train:end_train]).ravel()

# mu = np.min(mains_train, axis=0)
# sigma = np.std(mains_train, axis=0)
mu = 69
sigma = 450
mains_train -= mu
mains_train /= sigma
kettle_train /= sigma

# set window size
win_size = 128

# build input and output training data
num_examples = mains_train.size - win_size
X_train = np.zeros(shape=(num_examples, win_size, 1))
Y_train = np.zeros(shape=(num_examples, win_size, 1))

for i in xrange(0, num_examples):
    X_train[i, :, 0] = mains_train[i:i+win_size]
    Y_train[i, :, 0] = kettle_train[i:i+win_size]

# NN model

model_filename = 'kettle_model.h5'
filter_len = 4
num_filters = 8
if not os.path.isfile(model_filename):
    model = Sequential()
    model.add(Conv1D(input_shape=(win_size, 1), nb_filter=num_filters, filter_length=filter_len, border_mode='same', activation='linear'))
    model.add(Flatten())

    model.add(Dense(output_dim=win_size*num_filters, activation="relu"))
    model.add(Dense(output_dim=win_size, activation="relu"))
    model.add(Dense(output_dim=win_size*num_filters, activation="relu"))

    model.add(Reshape(target_shape=(win_size, num_filters)))
    model.add(Conv1D(nb_filter=1, filter_length=filter_len, border_mode='same', activation='relu'))

    model.compile(loss='mean_squared_error', optimizer=opt.adagrad(lr=0.01, epsilon=1e-8, decay=0.0),
                  metrics=['accuracy'])
    keras_plot(model, to_file='kettle_model.png', show_shapes=True, show_layer_names=True)
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=10, verbose=1, callbacks=[])

    model.save(model_filename)
else:
    model = load_model(model_filename)

pred = model.predict(X_train, batch_size=100, verbose=1)

mean_pred = np.zeros((mains_train.size,))
for i in xrange(0, num_examples):
    mean_pred[i:i+win_size] = mean_pred[i:i+win_size] + pred[i, :, 0]/float(win_size)

plt.figure()
plt.plot(Y_train[:, 0, 0])
plt.plot(mean_pred)
plt.show()



