import numpy as np

def limit_gpu_memory(per):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
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