import numpy as np
import pandas as pd
import argparse
import pickle

from malconv import Malconv
from preprocess import preprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser(description='Malconv-keras classifier')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default = 1)
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--limit', type=float, default = 0.)
parser.add_argument('--max_len', type=int, default = 200000)
parser.add_argument('--win_size', type=int, default = 500)
parser.add_argument('--val_size', type=float, default = 0.1, help="")
parser.add_argument('--save_path', type=str, default = 'saved')
parser.add_argument('--save_best', type=bool, default = True)
parser.add_argument('--csv', type=str)
args = parser.parse_args()

def train():
    if args.limit > 0:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.limit
        set_session(tf.Session(config=config))
    
    # data transform and padding
    data, label, tok = preprocess(args.csv, args.max_len, args.save_path)
    
    model = Malconv(args.max_len, args.win_size, len(tok.word_counts)+1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    # shuffle and split validation
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    split = int(len(data)*args.val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    
    # callbacks
    ear = EarlyStopping(monitor='val_acc', patience=5)
    mcp = ModelCheckpoint(os.path.join(args.save_path, 'malconv.h5'), 
                          monitor="val_acc", 
                          save_best_only=args.save_best, 
                          save_weights_only=False)
    
    history = model.fit(x_train, y_train,
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        verbose=args.verbose, 
                        callbacks=[ear, mcp],
                        validation_data=[x_test, y_test])

    
if __name__ == '__main__':
    train()