#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Perceptron Model
'''

import numpy as np

sign = lambda x: 0 if x < 0 else 1

training_data = [ 
    (np.array([3,3]), 1), 
    (np.array([4,3]), 1), 
    (np.array([1,1]), -1), 
]

# learning rate
eta = 1
b = 0
w = np.zeros(2)

def train_model():
    global b
    global w
    iters = 0
    while True:
        print 'Iters #', iters
        error_count = 0
        for data in training_data:
            res = np.dot(w, data[0]) + b
            # print predict_label
            if res * data[1] <= 0:
                error_count += 1
                w += eta * data[1] * data[0]
                b += eta * data[1]
                print w, b
        if 0 == error_count:
            break
        iters += 1

if __name__ == '__main__':
    train_model()