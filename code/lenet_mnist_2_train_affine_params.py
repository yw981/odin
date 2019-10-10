from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        for p in self.parameters():
            p.requires_grad = False
        self.apms = torch.nn.Parameter(torch.rand(size=(2, 3)))

    def forward(self, x):
        # 有问题，还是得用Siamese？
        # if x.size()[0] != 64:
        #     print(x.size())
        grid = F.affine_grid(self.apms.repeat((x.size()[0], 1, 1)), x.size())
        x = F.grid_sample(x, grid)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def my_loss(x, y, z, lm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff = z - torch.FloatTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to(device)

    # 0.1-小 0.2-94% 1.0-过大 加个过大的惩罚项？
    return F.nll_loss(x, y) - lm * torch.norm(diff, 2)


def train(model, device, train_loader, optimizer, epoch, lm):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print(data.size())
        # exit(1)
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = my_loss(output, target, model.apms, lm)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(model.apms)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    batch_size = 64
    torch.manual_seed(1234)
    epochs = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10000, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load('model/lenet_mnist_model.pth'))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    affine_params = []
    for lm in [0.01,0.05,0.1,0.12,0.14,0.17,0.2,0.22, 0.3, 0.4]:
        print(lm)
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, lm)
            print(model.apms)
            test(model, device, test_loader)
        affine_param = model.apms.cpu().detach().numpy()
        # print(affine_param.shape)
        affine_params.append(affine_param)

    affine_params = np.array(affine_params)
    print(affine_params.shape)
    print(affine_params)
    np.save('result/affine_params.npy', affine_params)


    # torch.save(model.state_dict(), "../model/lenet_mnist_affine_model.pth")


if __name__ == '__main__':
    main()
