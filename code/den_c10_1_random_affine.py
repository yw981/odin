from __future__ import print_function
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable

if __name__ == '__main__':
    CUDA_DEVICE = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    data, labels = next(iter(test_loader))
    labels = labels.numpy()
    np.random.seed(1234)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    epochs = 2

    model = torch.load('../../model/densenet10.pth')
    model = model.cuda(CUDA_DEVICE)
    v_data = Variable(data.cuda(CUDA_DEVICE), requires_grad=True)

    # in
    output = model.forward(v_data)
    output = F.softmax(output, dim=1)
    lg = output.data.cpu().numpy()
    amr = np.argmax(lg, axis=1)
    aml = labels
    wrong_indices = (amr != aml)
    right_indices = ~wrong_indices
    acc = (1 - np.sum(wrong_indices + 0) / aml.shape[0])
    np.save('../result/densenet_in.npy', lg)
    print('in saved ', acc)
    print(type(v_data))

    # 老版本？因为densenet引用了老版本，必须用回老板
    # device = torch.device("cuda" if use_cuda else "cpu")
    # data = data.to(device)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # model = Net().to(device)
    # model.load_state_dict(torch.load('model/lenet_mnist_model.pth'))
    k = 0
    params = []
    for i in range(9999):
        if i% 100 ==0 :
            print('it ' + str(i))

        tfparam = np.array([[1.1, 0., 0.], [0., 1.1, 0.]])
        tfseed = (np.random.rand(2, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6]])
        # print(tfseed)
        tfparam += tfseed
        # print(np.linalg.norm(tfparam))

        affine_param = torch.from_numpy(tfparam).float()
        # print(affine_param.size())
        p2 = data.size()
        # print(p2)
        # print(p2[0])
        p1 = affine_param.repeat(data.size()[0], 1, 1)

        # print(p1)

        grid = F.affine_grid(p1, p2)
        trans_data = F.grid_sample(data, grid)
        v_trans_data = Variable(trans_data.data.cpu().cuda(CUDA_DEVICE), requires_grad=True)
        output = model.forward(v_trans_data)
        output = F.softmax(output, dim=1)

        lg = output.data.cpu().numpy()

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

        if acc > 0.90:
            print("acc = %f" % acc)
            print('!!!!!!!! save #%d' % i)
            print(tfparam)
            params.append(tfparam)
            # np.save('result/exp_affine_in_%d.npy' % k, lg_softmax)
            print(lg.shape)
            k += 1
            if k >= 10:
                break

    params = np.array(params)
    print(params.shape)
    np.save('../result/affine_params_random.npy', params)
