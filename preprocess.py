import sys
import os
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def preprocess(csv_path, max_len, save_path=None):
    '''
    Return processed data and labels (ndarray)
    
    tokenizer.pkl, data.pkl, label.pkl would be saved
    '''
    print ('\nprocessing data ... need quite a while ......')
    df = pd.read_csv(csv_path, header=None)
    
    corpus = []
    for fn in df[0]:
        with open(fn, 'rb') as f:
            corpus.append(f.read())

    tok = Tokenizer(num_words=None,
                  filters='', 
                  lower=True,
                  split=' ', 
                  char_level=True,
                  oov_token="<unk>")
    
    print ('tokenizing...')
    tok.fit_on_texts(corpus)
    print ('transforming...')
    data = tok.texts_to_sequences(corpus)
    print ('padding...')
    seq = pad_sequences(data, maxlen=max_len, padding='post', truncating='post')
    
    with open(os.path.join(save_path, 'data.pkl'), 'wb') as f:
        pickle.dump(seq, f)
    with open(os.path.join(save_path, 'label.pkl'), 'wb') as f:
        pickle.dump(df[1].values, f)
    with open(os.path.join(save_path, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tok, f)
    
    return seq, df[1].values, tok



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('Usage: python preprocess.py <csv_path> <max_len> <save_path>')
    csv_path, max_len, path = sys.argv[1], sys.argv[2], sys.argv[3]
    data, label, tok = preprocess(csv_path, max_len, path)