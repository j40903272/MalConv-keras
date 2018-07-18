import numpy as np
from os.path import join
import pandas as pd
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

from malconv import Malconv
import utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

parser = argparse.ArgumentParser(description='Malconv-keras classifier training')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default = 1)
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--limit', type=float, default = 0.)
parser.add_argument('--max_len', type=int, default = 200000)
parser.add_argument('--win_size', type=int, default = 500)
parser.add_argument('--val_size', type=float, default = 0.1, help="")
parser.add_argument('--save_path', type=str, default = 'saved/')
parser.add_argument('--save_best', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('csv', type=str)


def train(model, max_len=200000, batch_size=64, verbose=True, epochs=100, save_path='saved/', save_best=True):
    
    # callbacks
    ear = EarlyStopping(monitor='val_acc', patience=5)
    mcp = ModelCheckpoint(join(save_path, 'malconv.h5'), 
                          monitor="val_acc", 
                          save_best_only=save_best, 
                          save_weights_only=False)
    
    history = model.fit_generator(
                            utils.data_generator(x_train, y_train, max_len, batch_size, shuffle=True),
                            steps_per_epoch=len(x_train)//batch_size + 1,
                            epochs=epochs, 
                            verbose=verbose, 
                            callbacks=[ear, mcp],
                            validation_data=utils.data_generator(x_test, y_test, max_len, batch_size),
                            validation_steps=len(x_test)//batch_size + 1 ,)
    return history

    
if __name__ == '__main__':
    args = parser.parse_args()
    
    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    
    # prepare model
    if args.resume:
        model = load_model(join(args.save_path, 'malconv.h5'))
    else:
        model = Malconv(args.max_len, args.win_size)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    
    # prepare data
    # preprocess is handled in utils.data_generator
    df = pd.read_csv(args.csv, header=None)
    data, label = df[0].values, df[1].values
    x_train, x_test, y_train, y_test = utils.train_test_split(data, label, args.val_size)
    
    
    history = train(model, args.max_len, args.batch_size, args.verbose, args.epochs, args.save_path, args.save_best)
    with open(join(args.save_path, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)