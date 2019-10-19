import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import roc_curve, auc
import torch
import torchvision
from torchvision import datasets, transforms
from func import arr_stat

RESULT_DIR = '../result_min'


def show_images(npdata, num_row=10):
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    rows = npdata.shape[0] // num_row + 1  # 加一行对比
    for i in range(npdata.shape[0]):
        img = npdata[i].reshape([28, 28])
        posp = fig.add_subplot(rows, num_row, i + 1)
        # plt.title('%d : %d -> %d %.2f' % (i, labels[i], error_preds[i], error_scores[i]))
        posp.imshow(img, cmap=plt.cm.gray)

    for i in range(num_row):
        dif = npdata[i + num_row].reshape([28, 28]) - npdata[i].reshape([28, 28])
        posp = fig.add_subplot(rows, num_row, num_row * 2 + i + 1)
        posp.imshow(dif, cmap=plt.cm.gray)

    plt.show()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def tpr95(in_data, out_data, is_diff=False):
    # calculate the falsepositive error when tpr is 95%
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))
    gap = (end - start) / 100000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr / total

    return fprBase


def auroc(in_data, out_data, is_diff=False):
    # calculate the AUROC

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 100000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    return aurocBase


def auprIn(in_data, out_data, is_diff=False):
    # calculate the AUPR

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 100000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(in_data, out_data, is_diff=False):
    # calculate the minimum detection error
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 100000
    # print(out_data.shape)
    # arr_stat('out data ', out_data)
    # 原著只有最高分进去比了，此处已修改
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    errorBase = 1.0

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    return errorBase


def compare_in_softmax_score_diff(tag, softmax_result, origin, labels):
    softmax_result = softmax_result.squeeze()
    # print5('ori', origin)
    # print5('trans', softmax_result)
    # print(softmax_result.shape)
    # 先算origin的，原始未经变换的结果
    origin_max_scores = np.max(origin, axis=1)
    origin_max_indices = np.argmax(origin, axis=1)
    # origin_max_indices = np.reshape(origin_max_indices,(softmax_result.shape[0],1))
    # print('origin_max_scores')
    # print5(origin_max_scores)
    # print(origin_max_indices)
    # print(origin_max_indices.shape)
    # 对应的得分
    cor_scores = softmax_result[tuple(np.arange(softmax_result.shape[0])), tuple(origin_max_indices)]
    # max_scores = np.max(softmax_result, axis=1)
    # max_indices = np.argmax(softmax_result, axis=1)
    # print('cor', cor_scores.shape)
    # print5(cor_scores)


    # 原计划，按逻辑diff_score = origin_max_scores - cor_scores，分数越低（可为负的）越好（越in），为计算AUROC取反了
    diff_score = cor_scores - origin_max_scores
    # print('diff_score')
    # print5('diff',diff_score)
    # arr_stat('diff', diff_score[0:5, ...])
    # arr_stat('diff ' + tag, diff_score)

    return diff_score


def affine_detector(tag, origin, variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """
    # 只有1和0的结果
    re = np.ones(origin.shape[0]).astype(np.bool)
    for variant in variant_arr:
        tfs = (np.argmax(origin, axis=1) == np.argmax(variant, axis=1))
        re = re & tfs

    # 输出都没变的
    print(tag, np.sum(re), ' out of ', re.shape[0], 'remain same, at rate ', np.sum(re) / re.shape[0])
    # return re


def cal_pr(y_score, y_test):
    # print(y_test)
    # print(y_score)

    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()


def cal_roc(y_score, y_test):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area，Numpy ravel()相当于flatten，内存引用不同
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # tpr fpr都是dict，0,1,2，micro
    # print(tpr)
    # print(fpr)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def vis_diff(tag, in_origin, in_variant_arr, out_origin, out_variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """

    # 只有1和0的结果
    # re = np.ones(origin.shape[0]).astype(np.bool)
    # for variant in variant_arr:
    #     tfs = (np.argmax(origin, axis=1) == np.argmax(variant, axis=1))
    #     re = re & tfs

    # 连续得分的结果，不变得分，变了不得分
    in_re = np.zeros(in_origin.shape[0])
    length = len(in_variant_arr)
    for variant in in_variant_arr:
        tfs = (np.argmax(in_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        in_re = in_re + tfs

    # print(in_re)

    # 连续得分的结果，不变得分，变了不得分
    out_re = np.zeros(out_origin.shape[0])
    length = len(out_variant_arr)
    for variant in out_variant_arr:
        tfs = (np.argmax(out_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        out_re = out_re + tfs

    # print(out_re)
    # print(in_re.shape)
    # print(out_re.shape)


    in_label = np.ones(in_re.shape)
    out_label = np.zeros(out_re.shape)

    y_label = np.hstack((in_label, out_label))
    re = np.hstack((in_re, out_re))
    # print(re.shape)
    # print(y_label.shape)
    cal_pr(re, y_label)


def vis_diff_roc(tag, in_origin, in_variant_arr, out_origin, out_variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """

    # 连续得分的结果，不变得分，变了不得分
    in_re = np.zeros(in_origin.shape[0])
    length = len(in_variant_arr)
    for variant in in_variant_arr:
        tfs = (np.argmax(in_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        in_re = in_re + tfs

    # print(in_re)

    # 连续得分的结果，不变得分，变了不得分
    out_re = np.zeros(out_origin.shape[0])
    length = len(out_variant_arr)
    for variant in out_variant_arr:
        tfs = (np.argmax(out_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        out_re = out_re + tfs

    # print(out_re)
    # print(in_re.shape)
    # print(out_re.shape)


    in_label = np.ones(in_re.shape)
    out_label = np.zeros(out_re.shape)

    y_label = np.hstack((in_label, out_label))
    re = np.hstack((in_re, out_re))
    # print(re.shape)
    # print(y_label.shape)
    cal_roc(re, y_label)


def calculate_out_diff_scores(tag, origin_out_result):
    out_diff_scores = []

    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_imagenet_%d.npy' % i)
        out_diff_score = compare_in_softmax_score_diff('imagenet %d' % i, result, origin_out_result, labels)
        out_diff_scores.append(out_diff_score)

    return np.array(out_diff_scores)


def evaluate_diff_score(tag, in_data, out_data, is_diff=False):
    # print(tag, ' fpr at tpr95 ', tpr95(in_data, out_data))
    # print(tag, ' error ', detection(in_data, out_data))
    # print(tag, ' AUROC ', auroc(in_data, out_data))
    # print(tag, ' AUPR in ', auprIn(in_data, out_data))
    str = "{} error : {:8.2f}% FPR at TPR95 : {:8.2f}% AUROC : {:>8.2f}% AUPR in : {:>8.2f}% "
    print(str.format(tag,
                     detection(in_data,
                               out_data,
                               is_diff) * 100,
                     tpr95(in_data,
                           out_data,
                           is_diff) * 100,
                     auroc(in_data,
                           out_data,
                           is_diff) * 100,
                     auprIn(in_data,
                            out_data,
                            is_diff) * 100))


# 测试
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
    compare_in_softmax_score_diff('in', origin_in_result, origin_in_result, labels)
    in_diff_scores = []
    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_in_%d.npy' % i)
        in_diff_score = compare_in_softmax_score_diff('cifar10 %d' % i, result, origin_in_result, labels)
        in_diff_scores.append(in_diff_score)

    in_diff_scores = np.array(in_diff_scores)

    # out_diff_scores = []
    # origin_out_result = np.load('../result/densenet_imagenet.npy')
    # for i in range(10):
    #     result = np.load('../result/densenet_imagenet_%d.npy' % i)
    #     out_diff_score = compare_in_softmax_score_diff('imagenet %d' % i, result, origin_out_result, labels)
    #     out_diff_scores.append(out_diff_score)



    for key in ['imagenet', 'gaussian', 'uniform']:
        origin_out_result = np.load(RESULT_DIR + '/densenet_%s.npy' % key)

        # arr_stat('in', origin_in_result)
        # arr_stat('out', origin_out_result)

        evaluate_diff_score('Baseline ', origin_in_result, origin_out_result)

        out_diff_scores = calculate_out_diff_scores(key, origin_out_result)

        for i in range(10):
            evaluate_diff_score('%s %d' % (key, i), in_diff_scores[i], out_diff_scores[i], True)
