#!/usr/bin/env python
# coding: utf-8



from random import shuffle
import torch


class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.indexes = list(range(len(self.data)))
        self.shuffle = shuffle
    def __iter__(self):
        self.start = 0
        if self.shuffle is True:
            shuffle(self.indexes)
        return self
    def __next__(self):
        start = self.start
        self.end = min(start+self.batch_size, len(self.data))
        if start >= self.end:
            raise StopIteration
        self.start += self.batch_size
        index = self.indexes[start:self.end]
        return self.data[index], self.labels[index]


def loss(y_hat, y):
    return torch.mean((y_hat.reshape(y.shape) - y)**2 / 2)


def sgd(params, lr, batch_size):
    with torch.no_grad():
        if isinstance(params, list):
            for i in range(len(params)):
                params[i] -= lr * params[i].grad / batch_size
                params[i].grad.zero_()
        else:
            params -= lr * params.grad / batch_size
            params.grad.zero_()

def relu(x):
    a = torch.zeros(x.shape)
    return torch.max(a, x)

def softmax(x):
    a = torch.exp(x)
    return a / torch.sum(a, dim=-1, keepdim=True)

def cross_entropy(y_hat, y):
    return -torch.sum((y*torch.log(y_hat)))

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

