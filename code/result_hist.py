import numpy as np
from func import arr_stat,RESULT_DIR
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from den_c10_3_2_score_diff import calculate_softmax_score_diff, calculate_out_diff_scores


def plot_two_hist(data1, color1, data2, color2, title='Title'):
    num_bins = 100
    # the histogram of the data
    plt.hist(data1, num_bins, facecolor=color1, alpha=0.5)
    plt.hist(data2, num_bins, facecolor=color2, alpha=0.5)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    _, labels = next(iter(test_loader))
    labels = labels.numpy()

    origin_in_result = np.load(RESULT_DIR + '/densenet_in.npy')

    # ------------- in 样本 对错的区分
    # arr_stat('in', origin_in_result)
    #
    # amr = np.argmax(origin_in_result, axis=1)
    # aml = labels
    # right_indices = (amr == aml)
    # # wrong_indices = ~wrong_indices
    # base_acc = (1 - np.sum(~right_indices + 0) / aml.shape[0])
    # print(base_acc)
    #
    # right_max_scores = np.max(origin_in_result, axis=1)[right_indices]
    # wrong_max_scores = np.max(origin_in_result, axis=1)[~right_indices]
    #
    # arr_stat('right', right_max_scores)
    # arr_stat('wrong', wrong_max_scores)
    # plot_two_hist(right_max_scores, 'blue', wrong_max_scores, 'red', 'right wrong')
    # in 样本 对错的区分 --------------------------end

    # ------------- in out样本的区分
    # in_max_scores = np.max(origin_in_result, axis=1)
    # for key in ['imagenet', 'gaussian', 'uniform']:
    #     origin_out_result = np.load(RESULT_DIR + '/densenet_%s.npy' % key)
    #     out_max_scores = np.max(origin_out_result, axis=1)
    #     plot_two_hist(in_max_scores, 'blue', out_max_scores, 'red', 'in out origin {} Baseline'.format(key))
    # in out样本的区分 --------------------------end

    # ------------- 变换后in out 区分
    origin_in_result = np.load(RESULT_DIR + '/densenet_in.npy')
    in_diff_scores = []
    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_in_%d.npy' % i)
        in_diff_score = calculate_softmax_score_diff('cifar10 %d' % i, result, origin_in_result)
        in_diff_scores.append(in_diff_score)

    in_diff_scores = np.array(in_diff_scores)

    for key in ['imagenet', 'gaussian', 'uniform']:
        origin_out_result = np.load(RESULT_DIR + '/densenet_%s.npy' % key)

        out_diff_scores = calculate_out_diff_scores(key, origin_out_result)

        plot_two_hist(np.reshape(in_diff_scores, (in_diff_scores.shape[0] * in_diff_scores.shape[1],)), 'blue',
                      np.reshape(out_diff_scores, (out_diff_scores.shape[0] * out_diff_scores.shape[1],)), 'red',
                      title='out diff %s ' % key)
    # 变换后in out 区分 --------------------------end
