#!/usr/bin/env python
# encoding: utf-8



class LmParams(object):

    tf_random_seed = 4862
    batch_size = 30
    init_weight = 0.2
    hidden_size = 256
    learn_rate = 0.0001
    max_gradient_norm = 5
    dropout_rate = 1
    max_epoch = 20
    class_num = 1
    layer_num = 2
    timestep_size = 7