from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import numpy as np
import time


def test_model(model, criterion, device, test_loader, tag='Test'):
    print('Process ', tag, )

    model.eval()
    t0 = time.time()
    idx = 0
    correct = 0

    for data, target in test_loader:
        # 为适配python3.5做处理
        data = Variable(data.cuda(0), requires_grad=True)

        outputs = model(data)
        # outputs = torch.tensor(
        #     [[-1.2211, -0.9080, 1.4694, 17.9587, -2.6866, 0.1167, -0.6995, -4.0835, -3.9281, -6.0281]]).to(device)
        # outputs.requires_grad_()
        nn_output = outputs.data.cpu().numpy()
        preds = np.argmax(nn_output, axis=1)  # get the index of the max log-probability
        correct_prediction = np.sum(np.equal(preds, target.cpu().numpy()))

        correct += correct_prediction

        if idx % 10 == 0:
            # print(target)
            total_test = (idx + 1) * data.size()[0]
            print(idx, ' ', correct, '/', total_test, ' acc ', correct / total_test)
        idx += 1

    print('{} set: Accuracy: {}/{} ({:.0f}%), {:.1f} seconds used.\n'.format(
        tag,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        time.time() - t0)
    )


if __name__ == '__main__':
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cuda'
    criterion = torch.nn.CrossEntropyLoss()

    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 100
    num_worker = 4

    # 模型！
    net = torch.load("../../model/{}.pth".format('densenet10'))
    # optimizer1 = optim.SGD(net1.parameters(), lr=0, momentum=0)
    net.cuda(0)

    # densenet 自训练 cifar10
    # model_path = '../model/densenet121_cifar.pth'
    # net = DenseNet121().to(device)
    # Test set: Average loss: 0.0010, Accuracy: 9516 / 10000(95 %)

    # vgg16 自训练 cifar10
    # model_path = '../model/vgg16_cifar10.pth'
    # net = VGG('VGG16').to(device)
    # Test set: Average loss: 0.0015, Accuracy: 9337 / 10000(93 %)

    # if use_cuda:
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    # checkpoint = torch.load(model_path)
    # net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    # print(model_path, ' epoch ', start_epoch, ' acc ', best_acc)

    # cifar10 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    # 若不Normalize Test set: Average loss: 0.0063, Accuracy: 7169 / 10000(72 %)
    data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_worker
    )

    test_model(net, criterion, device, data_loader, 'In ')
