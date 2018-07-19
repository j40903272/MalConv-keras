import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from preprocess import preprocess

def limit_gpu_memory(per):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per
    set_session(tf.Session(config=config))

def train_test_split(data, label, val_size=0.1):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    split = int(len(data)*val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    return x_train, x_test, y_train, y_test

def data_generator(data, labels, max_len = 200000, batch_size=64, shuffle=True):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    while True:
        for i in batches:
            xx = preprocess(data[i], max_len)[0]
            yy = labels[i]
            yield (xx, yy)