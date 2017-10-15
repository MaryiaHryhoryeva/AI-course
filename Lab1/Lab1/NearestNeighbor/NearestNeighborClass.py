'''
Created on Dec 1, 2016

@author:  
Adopted from CS231n
'''

import numpy as np
import progressbar
import heapq
import operator

class NearestNeighborClass(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y


    def cross_val(self,T1_im, T1_lab, T2_im, T2_lab, T3_im, T3_lab):
        k = [1, 2, 3, 4, 5]
        accur = []
        for i in k:
            self.Xtr = T1_im
            self.Xtr = np.concatenate((self.Xtr,T2_im))
            self.ytr = T1_lab
            self.ytr = np.concatenate((self.ytr,T2_lab))
            Lab_pred = self.predict(T3_im,i)
            accur.append(np.mean(T3_lab == Lab_pred))
            self.Xtr = T1_im
            self.Xtr = np.concatenate((self.Xtr, T3_im))
            self.ytr = T1_lab
            self.ytr = np.concatenate((self.ytr, T3_lab))
            Lab_pred = self.predict(T2_im, i)
            accur[i-1] += np.mean(T2_lab == Lab_pred)
            self.Xtr = T2_im
            self.Xtr = np.concatenate((self.Xtr, T3_im))
            self.ytr = T2_lab
            self.ytr = np.concatenate((self.ytr, T3_lab))
            Lab_pred = self.predict(T1_im, i)
            accur[i-1] += np.mean(T1_lab == Lab_pred)
        print "accur: " + str(accur)
        return accur.index(max(accur))+1


    def predict(self, X, k):

        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        #bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            #L2 distance:
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            minDist = np.argsort(distances)
            minDist = minDist[0:k]
            minDist = self.ytr[minDist]
            arr = np.zeros(10)
            for j in range(k):
                for jj in range(10):
                    if minDist[j]==jj:

                        arr[jj] += 1
            v, pred = arr.max(0),arr.argmax(0)
            Ypred[i]=pred
            #min_index = np.argmin(distances) # get the index with smallest distance
            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

            #bar.update(i+1)

        #bar.finish()

        return Ypred


