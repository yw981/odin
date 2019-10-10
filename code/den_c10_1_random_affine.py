from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


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


if __name__ == '__main__':
    print('hello')
    print('hello2222')

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10000)
    data, labels = next(iter(test_loader))
    labels = labels.numpy()
    np.random.seed(1234)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    epochs = 2

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    model.load_state_dict(torch.load('model/lenet_mnist_model.pth'))
    k = 0
    params = []
    for i in range(999):
        print('it ' + str(i))

        tfparam = np.array([[1.1, 0., 0.], [0., 1.1, 0.]])
        tfseed = (np.random.rand(2, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6]])
        # print(tfseed)
        tfparam += tfseed
        print(np.linalg.norm(tfparam))

        affine_param = torch.from_numpy(tfparam).to(device).float()
        grid = F.affine_grid(affine_param.repeat((data.size()[0], 1, 1)), data.size())
        trans_data = F.grid_sample(data, grid)
        output = model.forward(data.to(device))
        output = F.softmax(output, dim=1)

        lg = output.cpu().detach().numpy()

        # print(lg.shape)
        # print(mnist.test.labels.shape)
        amr = np.argmax(lg, axis=1)
        # print(amr)
        # aml = np.reshape(mnist.test.labels, (10000,1))
        aml = labels
        # print(aml)
        wrong_indices = (amr != aml)
        # print(wrong_indices)
        right_indices = ~wrong_indices
        acc = (1 - np.sum(wrong_indices + 0) / aml.shape[0])
        print("acc = %f" % acc)
        if acc > 0.95:
            print('save #%d' % i)
            print(tfparam)
            params.append(tfparam)
            # np.save('result/exp_affine_in_%d.npy' % k, lg_softmax)
            print(lg.shape)
            k += 1
            if k >= 10:
                break

    params = np.array(params)
    print(params.shape)
    np.save('result/affine_params_random.npy', params)
