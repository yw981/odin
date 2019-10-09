import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import numpy as np
# import time
# from scipy import misc

net1 = torch.load("../model/densenet10.pth")
optimizer1 = optim.SGD(net1.parameters(), lr=0, momentum=0)
net1.cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

inputs, _ = next(iter(testloaderIn))
outputs = net1(inputs)

# Calculating the confidence of the output, no perturbation added here
nnOutputs = outputs.data.cpu()
nnOutputs = nnOutputs.numpy()

print(nnOutputs)