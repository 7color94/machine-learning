#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from scipy.linalg import norm
import random
from matplotlib import pyplot
import random

def kmeans(X, k, observer=None, threshold=1e-15, maxiter=300):
    # X: 1450 * 2
    N = len(X)
    labels = np.zeros(N, dtype=int)
    centers = np.array(random.sample(X, k))
    iter = 0

    def calc_J():
        sum = 0
        for i in xrange(N):
            sum += norm(X[i]-centers[labels[i]])
        return sum

    def distmat(X, Y):
        n = len(X)
        m = len(Y)
        xx = np.sum(X*X, axis=1)
        yy = np.sum(Y*Y, axis=1)
        xy = np.dot(X, Y.T)

        print 'xx shape:', xx.shape
        print 'yy shape:', yy.shape
        print 'xy.shape:', xy.shape
        return np.tile(xx, (m, 1)).T + np.tile(yy, (n, 1)) - 2*xy

    Jprev = calc_J()
    while True:
        # notify the observer
        if observer is not None:
            observer(iter, labels, centers)

        # calculate distance from x to each center
        # distance_matrix is only available in scipy newer than 0.7
        # dist = distance_matrix(X, centers)
        dist = distmat(X, centers)
        # assign x to nearst center
        labels = dist.argmin(axis=1)
        # re-calculate each center
        for j in range(k):
            idx_j = (labels == j).nonzero()
            centers[j] = X[idx_j].mean(axis=0)

        J = calc_J()
        iter += 1

        if Jprev-J < threshold:
            break
        Jprev = J
        if iter >= maxiter:
            break

    # final notification
    if observer is not None:
        observer(iter, labels, centers)

if __name__ == '__main__':
    with open('cluster.pkl') as inf:
        samples = pickle.load(inf)

    N = 0
    for smp in samples:
        N += len(smp[0])

    X = np.zeros((N, 2))
    idxfrm = 0
    for i in range(len(samples)):
        idxto = idxfrm + len(samples[i][0])
        X[idxfrm:idxto, 0] = samples[i][0]
        X[idxfrm:idxto, 1] = samples[i][1]
        idxfrm = idxto

    def observer(iter, labels, centers):
        print "iter %d." % iter
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pyplot.plot(hold=False)  # clear previous plot
        pyplot.hold(True)

        # draw points
        data_colors=[colors[lbl] for lbl in labels]
        pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        # draw centers
        pyplot.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)

        pyplot.savefig('iter_%02d.png' % iter, format='png')

    kmeans(X, 3, observer=observer)