#!/usr/env/python

import pickle

import numpy as np
from numpy import testing

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import identity, softmax

import ctc_cost
from scribe import print_slab, prepare_print_pred

floatX = theano.config.floatX

theano.config.compute_test_value = 'off'
#theano.config.on_unused_input = 'ignore'

class Network:

    x = T.itensor3('inputs')     # num_batch, input_seq_len, num_inputs
    y = T.imatrix('targets')     # num_batch, output_seq_len

    mask_x = T.imatrix('mask_inputs')
    mask_y = T.imatrix('mask_targets')

    def __init__(self, num_batch, input_seq_len, num_inputs, output_seq_len, num_classes):
        self.num_batch = num_batch
        self.input_seq_len = input_seq_len
        self.num_inputs = num_inputs
        self.num_features = 15
        self.num_units = 50
        self.output_seq_len = output_seq_len
        self.num_classes = num_classes
        self.num_outputs = num_classes + 1  # add blank

        self.setup()

    def setup(self):
        # setup Lasagne Recurrent network
        # The output from the network is shape
        #  a) output_lin_ctc is the activation before softmax  (input_seq_len, batch_size, num_classes + 1)
        #  b) ouput_softmax is the output after softmax  (batch_size, input_seq_len, num_classes + 1)
        l_inp = InputLayer(shape=(self.num_batch, self.input_seq_len, self.num_inputs))
        l_mask = InputLayer(shape=(self.num_batch, self.input_seq_len))
        l_emb = EmbeddingLayer(l_inp, input_size=self.num_inputs, output_size=self.num_features)

        l_rnn = LSTMLayer(l_inp, num_units=self.num_units, peepholes=True, mask_input=l_mask)

        l_rnn_shp = ReshapeLayer(l_rnn, shape=(-1, self.num_units))
        l_out = DenseLayer(l_rnn_shp, num_units=self.num_outputs, nonlinearity=identity)
        l_out_shp = ReshapeLayer(l_out, shape=(-1, self.input_seq_len, self.num_outputs))

        # dimshuffle to shape format (input_seq_len, batch_size, num_classes + 1)
        #l_out_shp_ctc = lasagne.layers.DimshuffleLayer(l_out_shp, (1, 0, 2))

        l_out_softmax = NonlinearityLayer(l_out, nonlinearity=softmax)
        l_out_softmax_shp = ReshapeLayer(l_out_softmax, shape=(-1, self.input_seq_len, self.num_outputs))

        # calculate grad and cost
        output_lin_ctc = get_output(l_out_shp, {l_inp: self.x, l_mask: self.mask_x})
        output_softmax = get_output(l_out_softmax_shp, {l_inp: self.x, l_mask: self.mask_x})

        all_params = get_all_params(l_out_softmax_shp, trainable=True)  # dont learn embeddinglayer

        # the CTC cross entropy between y and linear output network
        pseudo_cost = ctc_cost.pseudo_cost(self.y, output_lin_ctc, self.mask_y, self.mask_x)

        # calculate the gradients of the CTC wrt. linar output of network
        pseudo_grad = T.grad(pseudo_cost.sum() / self.num_batch, all_params)
        true_cost = ctc_cost.cost(self.y, output_softmax, self.mask_y, self.mask_x)
        cost = T.mean(true_cost)

        shared_lr = theano.shared(lasagne.utils.floatX(0.001))
        #updates = lasagne.updates.sgd(pseudo_cost_grad, all_params, learning_rate=shared_lr)
        #updates = lasagne.updates.apply_nesterov_momentum(updates, all_params, momentum=0.9)
        updates = lasagne.updates.rmsprop(pseudo_grad, all_params, learning_rate=shared_lr)

        self.train = theano.function([self.x, self.mask_x, self.y, self.mask_y],
                                     [output_softmax, cost], updates=updates)
        self.test = theano.function([self.x, self.mask_x], [output_softmax])


if __name__ == '__main__':
    #Y_hat = np.asarray(np.random.normal(
    #    0, 1, (input_seq_len, num_batch, num_classes + 1)), dtype=floatX)
    #Y = np.zeros((target_seq_len, num_batch), dtype='int64')
    #Y[25:, :] = 1
    #Y_hat_mask = np.ones((input_seq_len, num_batch), dtype=floatX)
    #Y_hat_mask[-5:] = 0
    # default blank symbol is the highest class index (3 in this case)
    #Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    #X = np.random.random(
    #    (num_batch, input_seq_len)).astype('int32')
    #
    #y = T.imatrix('phonemes')
    #x = T.imatrix()   # batchsize, input_seq_len, features

    with open("digit.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)

    chars = data['chars']

    num_samples = len(data['x'])
    num_inputs = data['x'][0].shape[0]

    num_classes = len(chars)
    print_pred = prepare_print_pred(num_classes)

    max_input_seq_len = max([x.shape[-1] for x in data['x']])
    max_output_seq_len = max([len(y) for y in data['y']])

    #print(num_samples, num_inputs, num_classes, max_input_seq_len, max_output_seq_len)

    num_batch = 20
    net = Network(num_batch, max_input_seq_len, num_inputs, max_output_seq_len, num_classes)
    
    scale_to_int = 1024
    num_epoch = 100
    for epoch in range(num_epoch):
        print("\n## EPOCH", epoch)
        shuffle = np.random.permutation(num_samples)

        cost_lst = []
        for batch in range(num_samples//num_batch):
            idx = shuffle[batch*num_batch:(batch+1)*num_batch]
            
            X = np.zeros(shape=(num_batch, max_input_seq_len, num_inputs), dtype='int32')
            y = np.ones(shape=(num_batch, max_output_seq_len), dtype='int32') * num_classes
            mask_X = np.zeros(shape=(num_batch, max_input_seq_len), dtype=np.bool)
            mask_y = np.zeros(shape=(num_batch, max_output_seq_len), dtype=np.bool)

            for i in range(num_batch):
                j = idx[i]
                input_seq_len = data['x'][j].shape[-1]
                output_seq_len = len(data['y'][j])
                X[i, 0:input_seq_len] = data['x'][j].T * scale_to_int
                y[i, 0:output_seq_len] = np.array(data['y'][j])
                mask_X[i, 0:input_seq_len] = 1
                mask_y[i, 0:output_seq_len] = 1

            output_softmax, cost = net.train(X, mask_X, y, mask_y)
            cost_lst.append(cost)
            #testing.assert_almost_equal(pseudo_cost, pseudo_cost_old, decimal=4)
            #testing.assert_array_almost_equal(pseudo_cost_val, pseudo_cost_old_val)

        print("  - mean cost:", np.mean(cost_lst))

        for i in range(num_batch):
            j = idx[i]
            pred = np.argmax(output_softmax[i], axis=-1)
            pred = print_pred(pred)
            true = print_pred(y[i], ignore_repeat=True)
            print()
            print("target         :", true)
            print("prediction     :", pred)
            print("input bitmap   :")
            print_slab(data['x'][j])
            print("softmax firing :")
            input_seq_len = data['x'][j].shape[-1]
            print_slab(output_softmax[i, 0:input_seq_len].T)

