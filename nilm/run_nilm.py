import json, os, io
import numpy as np


def NN_conv1d(x, W, b, activation='linear'):
    # filter_len = W.shape[0]
    nb_filter_input = W.shape[2]
    nb_filter_output = W.shape[3]

    a = np.zeros((x.shape[0], nb_filter_output))
    for i in xrange(0, nb_filter_input):
        for j in xrange(0, nb_filter_output):
            a[:, j] += np.convolve(W[:, :, i, j].ravel(), x[:, i], mode='same') + b[j]

    if activation == 'linear':
        pass
    elif activation == 'relu':
        a = a * (a > 0.0)

    return a


def NN_flatten(x):
    a = x.ravel()

    return a


def NN_reshape(x, output_shape):
    a = np.reshape(x, output_shape)

    return a


def NN_dense(x, W, b, activation='linear'):
    a = np.dot(x, W) + b

    if activation == 'linear':
        pass
    elif activation == 'relu':
        a = a * (a > 0.0)

    return a


def disaggregate(x, npz_weights):
    x = x.reshape(x.size, 1)

    a0 = NN_conv1d(x, npz_weights['conv1_W'], npz_weights['conv1_b'], activation='linear')
    a1 = NN_flatten(a0)
    a2 = NN_dense(a1, npz_weights['dense1_W'], npz_weights['dense1_b'], activation='relu')
    a3 = NN_dense(a2, npz_weights['dense2_W'], npz_weights['dense2_b'], activation='relu')
    a4 = NN_dense(a3, npz_weights['dense3_W'], npz_weights['dense3_b'], activation='relu')
    a5 = NN_reshape(a4, a0.shape)
    a6 = NN_conv1d(a5, npz_weights['conv2_W'], npz_weights['conv2_b'], activation='relu')

    return a6


def get_measure_index(measure_index_filename):
    # read measure index log
    while 1:
        try:
            with open(measure_index_filename, 'r') as myfile:
                measure_index = int(myfile.read())
        except IOError as e:
            print e
            continue
        return measure_index


def get_measure_val(index, path='/home/root/measure/'):
    # read measurement value
    filename = path + 'mains' + str(index) + '.json'
    while 1:
        try:
            with open(filename, 'r') as myfile:
                data = json.load(myfile)
                val = data[data.keys()[0]]
        except IOError as e:
            print e
            continue
        return val

def get_measure_time(index, path='/home/root/measure/'):
    # read measurement value
    filename = path + 'mains' + str(index) + '.json'
    while 1:
        try:
            with open(filename, 'r') as myfile:
                data = json.load(myfile)
                timestamp = data.keys()[0]
        except IOError as e:
            print e
            continue
        return timestamp

def save_nilm_data(nilm_data, index, path='/home/root/nilm/kettle/'):
    # create JSON
    json_dict = dict(zip([get_measure_time(index)], nilm_data))

    # save JSON to file
    filename = path + 'kettle' + str(index) + '.json'
    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(json_dict, sort_keys=True, ensure_ascii=False)))

# measure files
measure_path = '/home/root/measure/'
measure_index_filename = measure_path + 'index.log'

# nilm path and index log
nilm_path = '/home/root/nilm/'
nilm_index_filename = nilm_path + 'index.log'

# restart from where it stopped, or reset
if os.path.exists(nilm_index_filename):
    with open(nilm_index_filename, 'r') as myfile:
        nilm_index = int(myfile.read())
else:
    nilm_index = 0

# get neural net weights
npz_weights = np.load(nilm_path + 'kettle_model.npz')

# normalization parameters
mu = 69.0
sigma = 450.0

# init vectors
win_size = npz_weights['dense2_b'].size
y = np.empty((win_size, 1))

# init buffers
buffers_path = nilm_path + 'buffers.npz'
if os.path.exists(buffers_path):
    buffers_npz = np.load(buffers_path)
    input_vec = buffers_npz['input_vec']
    output_vec = buffers_npz['output_vec']
else:
    input_vec = np.zeros((win_size, 1))
    output_vec = np.zeros((win_size, 1))
vecs = {'input_vec': input_vec, 'output_vec': output_vec}

# initial disaggregation
if nilm_index < win_size:

    for i in xrange(0, win_size):
        # rotate input and get a new sample
        input_vec[:win_size - 1] = input_vec[1:win_size]
        input_vec[-1] = (get_measure_val(i + 1) - mu) / sigma

        # disaggregate input data
        y = sigma * disaggregate(input_vec, npz_weights)

        # rotate output and get a new sample
        output_vec[:win_size - 1] = output_vec[1:win_size] + y[1:win_size]
        output_vec[-1] = y[-1]

        # output disaggregated data
        nilm_data = output_vec[0]/float(win_size)

        # save data
        save_nilm_data(nilm_data, i + 1)

        print "nilm_index = " + str(i)

    # save buffers
    vecs['input_vec'] = input_vec
    vecs['output_vec'] = output_vec
    np.savez('buffers', **vecs)

    # set nilm_index
    nilm_index = win_size
    with open(nilm_index_filename, 'w') as myfile:
        myfile.write(str(nilm_index))


while 1:

    # get index of not processed data
    measure_index = get_measure_index(measure_index_filename)
    num_samples = measure_index - nilm_index
    temp_index = nilm_index

    for i in xrange(temp_index, measure_index):

        # rotate input and get a new sample
        input_vec[:win_size-1] = input_vec[1:win_size]
        input_vec[-1] = (get_measure_val(i+1) - mu)/sigma

        # disaggregate input data
        y = sigma*disaggregate(input_vec, npz_weights)

        # rotate output and get a new sample
        output_vec[:win_size - 1] = output_vec[1:win_size] + y[1:win_size]
        output_vec[-1] = y[-1]

        # output disaggregated data
        nilm_data = output_vec[0]/float(win_size)

        # save data
        save_nilm_data(nilm_data, i + 1 - win_size)

        # save buffers
        vecs['input_vec'] = input_vec
        vecs['output_vec'] = output_vec
        np.savez(buffers_path, **vecs)

        # increment index and save to logfile
        nilm_index += 1
        print "nilm_index = " + str(nilm_index)
        with open(nilm_index_filename, 'w') as myfile:
            myfile.write(str(nilm_index))
