# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

import os
import argparse
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import numpy as np
from src.vdcnnIR import Vgg
from src.preprocess import preprocess
import matplotlib.pyplot as plt
import torch.optim as optim
import scikitplot
import shutil
def get_args():
    parser = argparse.ArgumentParser("""Very Deep Convolutional Networks for Large Scale Image Recognition""")
    parser.add_argument('-t', '--train', type=str, default='train', help="""required image dataset for training a model.
                                                                           It must be in the data directory """)
    parser.add_argument('-v', '--val', type=str, default='val', help="""required image dataset for training a model.
                                                                              It must be in the data directory """)
    parser.add_argument('-b', '--batchsize', type=int, choices=[64,128,256], default=128, help='select number of samples to load from dataset')
    parser.add_argument('-e', '--epochs', type=int, choices=[50, 100, 150], default=50)
    parser.add_argument('--d', '--depth', type=int, choices=[11,13,16,19], default=11, help='depth of the deep learning model')
    parser.add_argument('-c11', '--conv1_1', action='store_true', default=False,
                        help="""setting it True will replace some of the 3x3 Conv layers with 1x1 Conv layers in the 16 layer network""")
    parser.add_argument('--es', '--early_stopping', type=int, default= 6, help="""early stopping is used to stop training of network, 
                                                                        if does not improve validation loss""")

    parser.add_argument()





def train(opt):
    traindata, trainGenerator, classes = preprocess(path='./data'+os.sep+opt.train, batchsize=opt.batchsize, shuffle=True)
    valdata,validationGenerator, classes = preprocess(path='./data'+os.sep+opt.val, batchsize=opt.batchsize, shuffle=False)
    num_channels = iter(traindata).__next__().size()[1]
    path = 'results/VdcnnIR_{}.txt'.format(opt.depth)
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mknod(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Vgg(num_channels=num_channels,num_classes=classes,initialize_weights=True,conv1_1=opt.conv1_1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    def early_stopping(val_loss, model, patience= opt.early_stopping):
        early_stop = None
        count = 0
        best_score = None
        if best_score is None:
            best_score = val_loss
            torch.save(model, 'models/VdcnnIR_{}'.format(opt.depth))
        elif val_loss < best_score:
            count +=1
            print("Loss:{} doesn't improved from {}".format(val_loss, best_score))
            if count >= patience:
                early_stop = True
        else:
            print("Loss:{} improved from {}. Saving model........".format(val_loss, best_score))
            best_score = val_loss
            torch.save(model, 'models/VdcnnIR_{}'.format(opt.depth))
            count = 0
        return early_stop, best_score

    totalVal_loss = []
    totalTrain_loss = []
    for epoch in range(opt.epochs):
        model.train()
        train_loss = []
        total_predictions = []
        total_labels = []
        for idx, data in enumerate(traindata):
            data_, label = data.to(device)
            optimizer.zero_grad()
            prob = model.fit(data_).view(-1)
            prob_ = np.argmax(prob.detach().cpu(), -1)
            loss = criterion(prob, label)
            train_loss.append(loss.item()*len(label.cpu()))
            loss.backward()
            optimizer.step()
            total_predictions.extend(prob_)
            total_labels.extend(label.cpu())
            print('Iter[{}/{}]\tEpoch[{}/{}]\tLoss{}\tacc{}'.format(idx+1, len(trainGenerator), epoch+1, opt.epochs, loss.item(),metrics.accuracy_score(label.cpu(), prob_)))
        loss_epoch = sum(train_loss)/len(traindata)
        totalTrain_loss.append(loss_epoch)
        with open(path, 'a') as f:
            f.write('Epoch{}\tLoss{}\tAccuracy{}'.format(epoch+1, loss_epoch, metrics.accuracy_score(total_labels,total_predictions)))

        model.eval()
        val_loss = []
        total_Valpredictions = []
        total_ValLabels = []

        for idx_e, data_e in enumerate(validationGenerator):
            data_e,label_e = data_e.to(device)
            with torch.no_grad():
                prob_e = model(data_e)
                pred_e = np.argmax(prob_e,-1)
                loss = criterion(prob_e, label_e)
                val_loss.append(loss.item()*len(label_e))






