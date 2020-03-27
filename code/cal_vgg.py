# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function

import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import calData as d
import calMetric as m
import vgg1

# CUDA_DEVICE = 0

start = time.time()
# loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])

# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
# nnName = "densenet10"

# imName = "Imagenet"



criterion = nn.CrossEntropyLoss()


def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net1 = vgg1.VGG('VGG16').to(device)
    # Test set: Average loss: 0.0015, Accuracy: 9337 / 10000(93 %)

    if use_cuda:
        net1 = torch.nn.DataParallel(net1)
        cudnn.benchmark = True

    checkpoint = torch.load("../../model/{}.pth".format(nnName))
    net1.load_state_dict(checkpoint['net'])
    net1.eval()

    if dataName != "Uniform" and dataName != "Gaussian":
        testsetout = torchvision.datasets.ImageFolder("../../data/{}".format(dataName), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                                    shuffle=False, num_workers=2)

    # if nnName == "densenet10" or nnName == "wideresnet10":
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                               shuffle=False, num_workers=2)
    # if nnName == "densenet100" or nnName == "wideresnet100":
    #     testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform)
    #     testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                                shuffle=False, num_workers=2)

    if dataName == "Gaussian":
        d.testGaussian(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)

    elif dataName == "Uniform":
        d.testUni(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)
    else:
        d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)
