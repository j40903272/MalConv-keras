import utils
import numpy as np
import pandas as pd
import argparse

from keras.models import load_model

parser = argparse.ArgumentParser(description='Malconv-keras classifier')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default = 1)
parser.add_argument('--limit', type=float, default = 0.)
parser.add_argument('--model_path', type=str, default = 'saved/malconv.h5')
parser.add_argument('--result_path', type=str, default = 'saved/result.csv')
parser.add_argument('csv', type=str)

def predict(model, fn_list, label, batch_size=64, verbose=1):
    
    max_len = model.input.shape[1]
    pred = model.predict_generator(
                    utils.data_generator(fn_list, label, max_len, batch_size, shuffle=False),
                    steps=len(fn_list)//batch_size + 1,
                    verbose=verbose
                    )
    return pred

if __name__ == '__main__':
    args = parser.parse_args()
    
    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    # load model
    model = load_model(args.model_path)
    
    # read data
    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    label = np.zeros((fn_list.shape))
    
    pred = predict(model, fn_list, label, args.batch_size, args.verbose)
    df['predict score'] = pred
    df[0] = [i.split('/')[-1] for i in fn_list]
    df.to_csv(args.result_path, header=None, index=False)
    print ('Results writen in', args.result_path)