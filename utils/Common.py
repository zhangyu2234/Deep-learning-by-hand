#!/usr/bin/env python
# coding: utf-8

from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
    return -torch.log(y_hat[range(len(y_hat)), y])


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def show_img(data, figsizes, num_imgs=6):
    _, axes = plt.subplots(2, 3, figsize=figsizes)
    for i in range(num_imgs):
      ax = axes[i//3, i%3]
      img, label = data[i]
      ax.imshow(img.squeeze(), cmap='gray')
      ax.set_title(f'Label:{label}')
      ax.axis('off')


class ImageDataLoader:
    def __init__(self, datasets, batch_size, shuffle=False):
        self.datasets = datasets
        self.batch_size = batch_size
        self.indexes = list(range(len(self.datasets)))
        self.shuffle = shuffle
        self.num_imgs = len(datasets)

    def __iter__(self):
        self.start = 0
        if self.shuffle is True:
            shuffle(self.indexes)
        return self

    def __next__(self):
        start = self.start
        self.end = min(start+self.batch_size, len(self.datasets))
        if start >= self.end:
            raise StopIteration
        self.start += self.batch_size
        index = self.indexes[start:self.end]
        batch = [self.datasets[i] for i in index]
        data = torch.stack([img[0] for img in batch])
        label = torch.tensor([labels[1] for labels in batch])
        return data, label


def plot_loss_acc(loss, acc, figsize, num_epochs, xlabel1='Epoch', ylabel1='Loss', ylabel2='Accuracy', color1='red', color2='blue'):
  fig, ax1 = plt.subplots(figsize=figsize)
  ax1.set_xlabel(xlabel1)
  ax1.set_ylabel(ylabel1, color=color1)
  ax1.plot(range(1, num_epochs+1, 1), loss, color=color1)
  # Configure the style of tick marks on the axis
  ax1.tick_params(axis='y', labelcolor=color1)
  ax2 = ax1.twinx()
  ax2.set_ylabel(ylabel2, color=color2)
  ax2.plot(acc, color=color2)
  ax2.tick_params(axis='y', labelcolor=color2)

def validation(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, label in data_loader:
            y = model(imgs)
            _, predicted = torch.max(y, dim=1)
            correct += (predicted == label).sum()
            total += label.size(0)
    return correct / total
            
    
def train_and_test(num_epochs, model, loss_function, optim, valid, train, test, figsize):
  loss_value = []
  acc_value = []
  for epoch in range(num_epochs):
    batch_loss = []
    for i, (img, label) in enumerate(train):
      l = loss_function(model(img), label)
      batch_loss.append(l.item())
      optim.zero_grad()
      l.backward()
      optim.step()
      batch_loss.append(l.item())
      if (i+1) % 100 == 0:
        print(f'Step:{i+1}/{len(train)}, Epoch:{epoch+1}/{num_epochs}, Loss:{l.item():.4f}')
    avg_loss = sum(batch_loss) / len(batch_loss)
    loss_value.append(avg_loss)
    accuracy = valid(model, test)
    acc_value.append(accuracy)
    print(f'Step:{i+1}/{len(train)}, Epoch:{epoch+1}/{num_epochs}, Accuracy:{accuracy:.2f}')
  plot_loss_acc(loss_value, acc_value, figsize, num_epochs)
    
def validation_gpu(model, data_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, label in data_loader:
          imgs = imgs.to(device=device)
          label = label.to(device=device)
          y, _ = model(imgs)
          _, predicted = torch.max(y, dim=1)
          correct += (predicted == label).sum()
          total += label.size(0)
    return correct / total
    
def train_and_test_gpu(num_epochs, model, loss_function, optim, valid, train, test, device, figsize):
  loss_value = []
  acc_value = []
  for epoch in range(num_epochs):
    batch_loss = []
    for i, (img, label) in enumerate(train):
      img = img.to(device=device)
      label = label.to(device=device)
      l = loss_function(model(img), label)
      batch_loss.append(l.item())
      optim.zero_grad()
      l.backward()
      optim.step()
      batch_loss.append(l.item())
      if (i+1) % 100 == 0:
        print(f'Step:{i+1}/{len(train)}, Epoch:{epoch+1}/{num_epochs}, Loss:{l.item():.4f}')
    avg_loss = sum(batch_loss) / len(batch_loss)
    loss_value.append(avg_loss)
    accuracy = valid(model, test, device)
    # The data is stored on the GPU, first moved to the CPU, and then converted to Python scalars.
    acc_value.append(accuracy.item())

    print(f'Step:{i+1}/{len(train)}, Epoch:{epoch+1}/{num_epochs}, Accuracy:{accuracy:.2f}')
  plot_loss_acc(loss_value, acc_value, figsize, num_epochs)

