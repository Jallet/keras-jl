from __future__ import absolute_import
from . import backend as K
import numpy as np
from theano import *
import theano.tensor as T


class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {'name': self.__class__.__name__}


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0., ld = 0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        # self.ld = K.cast_to_floatx(ld)
        # self.ld = 0.1
        self.ld = K.cast_to_floatx(0.)
        self.l2 = K.cast_to_floatx(0.002)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        # print "type of p: ", type(self.p)
        # a = self.p.get_value()
        # print "type of a: ", type(a)
        # print "shape of a: ", a.shape
        
        regularized_loss = loss + K.sum(K.abs(self.p)) * self.l1
        regularized_loss += K.sum(K.square(self.p)) * self.l2
        # regularized_loss += K.sum(self.min_weight - self.p) * self.lw
        # return K.in_train_phase(regularized_loss, loss)
        
        # regularized_loss += K.sum(K.pow(self.p, 4)) * 1000
        p_array = self.p.get_value() 
        print "type of self.p: ", type(self.p)
        print "ndim of self.p: ", p_array.ndim
        print "shape of self.p: ", p_array.shape
        if 4 == self.p.ndim or 2 == self.p.ndim:
            p_array = self.p.get_value() 
            print "ndim of self.p: ", self.p.shape
            shape = self.p.shape
            size = self.p.size
            row = shape[0]
            col = size / row
            print "type of row: ", row.type, ", type of col: ", col.type
            if 4 == self.p.ndim:
                reshaped_p = K.reshape(self.p, (row, col))
            else:
                reshaped_p = K.transpose(self.p)
                row = reshaped_p.shape[0]
                col = reshaped_p.shape[1]
            p_mean = K.mean(reshaped_p, axis = 0, keepdims = True)
            p_std = K.std(reshaped_p, axis = 0, keepdims = True)
            # centered_p = (reshaped_p - p_mean)  / p_std
            centered_p = (reshaped_p - p_mean)
            
            centered_p_t = K.transpose(centered_p)
            covariance = K.dot(centered_p, centered_p_t) / col
            print "type of covariance: ", type(covariance)
            print "type of covariance: ", covariance.type
            mask = T.eye(row)
            # regularized_loss += (K.sum(K.square(covariance)) - K.sum(K.square(mask * covariance) - K.sum(mask * covariance))) * self.ld
            # regularized_loss += (K.sum(covariance) - 2 * K.sum(mask * covariance)) * self.ld
            regularized_loss += K.sum(K.square(covariance - mask * covariance)) * self.ld / (row- 1)
            # regularized_loss -= K.sum(mask * covariance) * self.ld
            # regularized_loss -= K.sum(K.square(centered_p)) * self.ld
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1,
                'l2': self.l2}


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            regularized_loss += self.l1 * K.sum(K.mean(K.abs(output), axis=0))
            regularized_loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1,
                'l2': self.l2}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)

def l1l2ld(l1 = 0.01, l2 = 0.01, ld = 0.01):
    print "l1l2ld"
    return WeightRegularizer(l1 = l1, l2 = l2, ld = ld)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)


from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
