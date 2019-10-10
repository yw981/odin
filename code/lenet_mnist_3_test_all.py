from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        # for p in self.parameters():
        #     p.requires_grad = False
        self.apms = torch.nn.Parameter(torch.rand(size=(2, 3)))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_save(model, device, data, label):
    output = model.forward(data.to(device))
    output = F.softmax(output, dim=1)
    np.save('result/%s.npy' % label, output.cpu().detach().numpy())
    print(label, ' saved')


def test_origin_and_affines(affine_params, model, device, data, tag):
    data = data.to(device)
    test_save(model, device, data, tag)
    for i in range(10):
        label = '%s_%d' % (tag, i)
        affine_param = torch.from_numpy(affine_params[i]).to(device).float()
        grid = F.affine_grid(affine_param.repeat((data.size()[0], 1, 1)), data.size())
        trans_data = F.grid_sample(data, grid)
        test_save(model, device, trans_data, label)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--affine', type=str, default='affine_params.npy', metavar='N',
                        help='affine array file name')

    args = parser.parse_args()
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    epochs = 2

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    model.load_state_dict(torch.load('model/lenet_mnist_model.pth'))
    affine_params = np.load('result/'+args.affine)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10000, **kwargs)
    data, _ = next(iter(test_loader))
    tag = 'lenet_in'
    test_origin_and_affines(affine_params, model, device, data, tag)

    trans = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='../data', train=False, transform=trans, download=True)
    bs = 10000
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
    data, _ = next(iter(data_loader))
    data = data[:, 0, :, :].view((bs, 1, 28, 28))
    # eval('cifar', data.numpy())
    tag = 'lenet_cifar'
    test_origin_and_affines(affine_params, model, device, data, tag)
    # test_save(model, device, data, label)
    #

    # for i in range(10):
    #     label = 'lenet_cifar_%d' % i
    #     affine_param = torch.from_numpy(affine_params[i]).to(device).float()
    #     grid = F.affine_grid(affine_param.repeat((data.size()[0], 1, 1)), data.size())
    #     trans_data = F.grid_sample(data, grid)
    #     test_save(model, device, trans_data, label)

    data = torch.from_numpy(np.random.uniform(size=(10000, 1, 28, 28))).float()
    tag = 'lenet_gaussian'
    test_origin_and_affines(affine_params, model, device, data, tag)


if __name__ == '__main__':
    main()
