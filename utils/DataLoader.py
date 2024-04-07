#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import shuffle


# In[2]:


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
            self.shuffle(self.indexes)
        return self
    def __next__(self):
        start = self.start
        self.end = min(start+self.batch_size, len(self.data))
        if start >= self.end:
            raise StopIteration
        self.start += self.batch_size
        index = self.indexes[start:self.end]
        return self.data[index], self.labels[index]





