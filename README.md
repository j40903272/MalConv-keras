# MalConv-keras
A Keras implementation of MalConv

---
## Desciprtion

This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435).

This model architecture can be used for any very long sequence classification.

## Requirement
- python3
- numpy
- pandas
- pickle
- keras (tensorflow)

## Get started
#### Clone the repository
```
git clone https://github.com/j40903272/MalConv-keras
```
#### Prepare data
Prepare a csv file with filenames and labels in the  **<filename**, **label>**  format
```
0778a070b283d5f4057aeb3b42d58b82ed20e4eb_f205bd9628ff8dd7d99771f13422a665a70bb916, 0
fbd1a4b23eff620c1a36f7c9d48590d2fccda4c2_cc82281bc576f716d9a0271d206beb81ad078b53, 0
```
see more in [exmaple.csv](https://github.com/j40903272/MalConv-keras/blob/master/example.csv)
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
The default path for output files would be in [saved/](https://github.com/j40903272/MalConv-keras/tree/master/saved)
