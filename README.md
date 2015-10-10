Yet another LSTM-CTC example for Lasagne/Theano
===============================================
This is not my own creation, composited from the following sources:

* http://github.com/rakeshvar/rnn_ctc.git
* http://github.com/skaae/Lasagne-CTC.git


## Install & Run

1. data generation
```
python script.py
```
The generated data filename is 'digits.pkl'.

2. training
```
python test_ctc.py
```
if you need to make a log file, use the following commands.
```
unbuffer python test_ctc.py 2>&1 | tee output.log
```

