# MalConv-keras
A Keras implementation of MalConv and adversarial sample

---
## Desciprtion

This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435) which can be used for any very long sequence classification.

The adversarial samples are crafted by padding some bytes to the input file. It would fail if the origin file length exceeds the model's input size.

Enjoy !

## Requirement
- python3 (3.5.2)
- numpy (1.13.1)
- pandas (0.22.0)
- pickle (0.7.4)
- keras (2.1.5)
- tensorflow (1.6.0)
- sklearn

## Get started
#### Clone the repository
```
git clone https://github.com/j40903272/MalConv-keras
```
#### Prepare data
Prepare a csv file with filenames(absolute or relative path) and labels in the  **<filename**, **label>**  format
```
0778a070b283d5f4057aeb3b42d58b82ed20e4eb_f205bd9628ff8dd7d99771f13422a665a70bb916, 0
fbd1a4b23eff620c1a36f7c9d48590d2fccda4c2_cc82281bc576f716d9a0271d206beb81ad078b53, 0
```
see more in [example.csv](https://github.com/j40903272/MalConv-keras/blob/master/example.csv) (1:benign, 0:malicious)
#### Training
```
python3 train.py example.csv
python3 train.py example.csv --resume
```
#### Predict
```
python3 predict.py example.csv
python3 predict.py example.csv --result_path saved/result.csv
```

#### Preprocess
If you require the preprocessed data, run the following
```
python3 preprocess.py example.csv
python3 preprocess.py example.csv --save_path saved/preprocess_data.pkl
```
#### Adversarial
Try different --step_size, it's quite sensitive
```
python3 gen_adversarial.py example.csv
python3 gen_adversarial.py example.csv --save_path saved/adversarial_samples --pad_percent 0.1

### for multiple class classification
python3 gen_adversarial2.py example.csv --class 1
```
The process log format would be **<filename**, **original score, file length, pad length, loss, predict score>**
as in [adversarial_log.csv](https://github.com/j40903272/MalConv-keras/blob/master/saved/adversarial_log.csv)

**< Notice >**
The generated padding bytes sometimes cannot be corrected encoded, a workaround is as follow :
```
# Read bytes then tokenize
byte_content = open('target', 'rb').read()
content = [chr(i) for i in byte_content]
```

#### Parameters
Find out more options with `-h`
```
python3 train.py -h

  -h, --help
  --batch_size BATCH_SIZE
  --verbose VERBOSE
  --epochs EPOCHS
  --limit LIMIT
  --max_len MAX_LEN
  --win_size WIN_SIZE
  --val_size VAL_SIZE
  --save_path SAVE_PATH
  --save_best
  --resume
  
python3 predict.py -h
python3 preprocess.py -h
```
#### Logs and checkpoint
The default path for output files would all be in [saved/](https://github.com/j40903272/MalConv-keras/tree/master/saved)

## Example
```
from malconv import Malconv
from preprocess import preprocess
import utils

model = Malconv()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

df = pd.read_csv(input.csv, header=None)
filenames, label = df[0].values, df[1].values
data = preprocess(filenames)
x_train, x_test, y_train, y_test = utils.train_test_split(data, label)

history = model.fit(x_train, y_train)
pred = model.predict(x_test)
```
