Yet another LSTM-CTC example for Lasagne/Theano
===============================================
This is not only my own creation. Most of the codes are originated from the following sources:

* http://github.com/rakeshvar/rnn_ctc.git
* http://github.com/skaae/Lasagne-CTC.git


## Install & Run

### data generation
```
python3 scribe.py
```
The generated data filename is `digit.pkl`.

### training
```
python3 test_ctc.py
```
if you need to make a log file, use the following commands.
```
unbuffer python3 test_ctc.py 2>&1 | tee output.log
```

