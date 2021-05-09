import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import pandocfilters as f
import os
import matplotlib.image as img

from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from glob import glob
from PIL import Image
from termcolor import colored
from sklearn.metrics import f1_score



class CactiDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name,label= self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = Image.open(img_path)
        image = image.resize((224, 224))
        if self.transform is not None:
            img = self.transform(image)

        return img, label


predap = []
targetap = []

transforms = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor()])

train_lab = pd.read_csv(r'agedatasetv4/train.csv')
train_path = r'agedatasetv4/Train'


# img_dir = '/content/drive/My Drive/data'
# train_data = datasets.ImageFolder(img_dir, transform=train_transforms)

2

train_data = CactiDataset(train_lab, train_path, transforms)

valid_size = 0.1

# convert data to a normalized torch.FloatTensor

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_split = int(np.floor((valid_size) * num_train))
valid_idx, train_idx = indices[:valid_split], indices[valid_split:]
print(len(valid_idx), len(train_idx))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                           sampler=train_sampler, num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                           sampler=valid_sampler, num_workers=0)


model = models.densenet121(pretrained=True)
# 12 30
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier.in_features

# model.classifier[1]=nn.Conv2d(512,2,kernel_size=(1,1), stride=(1,1))

model.classifier = nn.Linear(num_ftrs, 3)

fc_parameters = model.classifier.parameters()
for param in fc_parameters:
    param.requires_grad = True

model
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)


# %%


def train_accuracy(model, criterion, use_cuda, train_loss):
    # monitor test loss and accuracy
    # train_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        train_accuracy = 100. * correct / total
    print('\ntrain Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    return train_accuracy


def test(model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    pre = 0.0
    count = 0
    rec = 0.0
    f1_m = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        y_pred_list = torch.tensor([a.squeeze().tolist() for a in pred])
        targetlist = torch.tensor([t.squeeze().tolist() for t in target])
        # met=ConfusionMatrix()

        # print(y_pred_list)
        # print(targetlist)

        count = count + 1
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        test_accuracy = 100. * correct / total
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_accuracy


def valid_accuracy(model, criterion, use_cuda, valid_loss):
    # monitor test loss and accuracy
    # train_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(valid_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        val_accuracy = 100. * correct / total
    print('\nvlaidation Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return val_accuracy


def train(n_epochs, model, optimizer, criterion, use_cuda, save_path):
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()
            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # back prop
            loss.backward()
            # grad
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # test(model, criterion, use_cuda)
        tra_ac = train_accuracy(model, criterion, use_cuda, train_loss)
        val_acc = valid_accuracy(model, criterion, use_cuda, valid_loss)

        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss

    # return trained model

    return model


# %%

model = train(25, model, optimizer, criterion, use_cuda, 'Densenet121.pt')
