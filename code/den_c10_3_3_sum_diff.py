import numpy as np
from func import arr_stat, RESULT_DIR


def tpr95(in_data, out_data, is_diff=False):
    return 0
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
        if tpr >= 0.9495:
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


def calculate_softmax_score_diff(tag, softmax_result, origin):
    softmax_result = softmax_result.squeeze()

    # 先算origin的，原始未经变换的结果
    origin_max_scores = np.max(origin, axis=1)
    origin_max_indices = np.argmax(origin, axis=1)
    # 对应的得分
    cor_scores = softmax_result[tuple(np.arange(softmax_result.shape[0])), tuple(origin_max_indices)]

    # 原计划，按逻辑diff_score = origin_max_scores - cor_scores，分数越低（可为负的）越好（越in），为计算AUROC取反了
    # 统一取绝对值的相反数，最大0最好（越in），越小越out
    diff_score = -np.abs(cor_scores - origin_max_scores)

    return diff_score


def calculate_out_diff_scores(tag, origin_out_result):
    out_diff_scores = []

    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_imagenet_%d.npy' % i)
        out_diff_score = calculate_softmax_score_diff('imagenet %d' % i, result, origin_out_result)
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


# 求和区分，图像经过所有变换后，所有变换的变换值求和累积，再对比计算
if __name__ == "__main__":
    L = 10

    origin_in_result = np.load(RESULT_DIR + '/densenet_in.npy')
    print(origin_in_result.shape)
    # calculate_softmax_score_diff('in', origin_in_result, origin_in_result)
    in_diff_scores = []
    in_results = []
    for i in range(L):
        in_result = np.load(RESULT_DIR + '/densenet_in_%d.npy' % i)
        # 目前版本值中已经softmax过，待改
        in_results.append(in_result)

    in_results = np.array(in_results)
    # print(in_results.shape)
    in_max_scores = np.max(origin_in_result, axis=1)
    # arr_stat('in max ', in_max_scores)
    # print(in_max_scores[:20])
    in_max_indices = np.argmax(origin_in_result, axis=1)

    in_sum_score = np.zeros(in_max_scores.shape)
    # print(origin_max_indices)
    for i in range(L):
        # print(in_results[i].shape)
        # print(np.sum(in_results[i][0]))

        # 先算origin的，原始未经变换的结果

        # 对应的得分
        cor_scores = in_results[i][tuple(np.arange(in_results[i].shape[0])), tuple(in_max_indices)]
        diff_score = np.abs(cor_scores - in_max_scores)
        # diff_score = cor_scores - in_max_scores
        # print('in ',diff_score[0:5])
        in_sum_score += diff_score

    # print('in sum ', in_sum_score[:10])
    arr_stat('in sum', in_sum_score)
    # print(in_sum_score.shape)
    # dsg0 = sum_score[sum_score > 0]
    # print('in ',i)
    # print(dsg0.shape)
    # print(np.sum(dsg0))
    # dsl0 = sum_score[sum_score < 0]
    # print(dsl0.shape)
    # print(np.sum(dsl0))

    # 原计划，按逻辑diff_score = origin_max_scores - cor_scores，分数越低（可为负的）越好（越in），为计算AUROC取反了
    # 统一取绝对值的相反数，最大0最好（越in），越小越out
    #
    #
    #     in_diff_score = calculate_softmax_score_diff('cifar10 %d' % i, result, origin_in_result)
    #     in_diff_scores.append(in_diff_score)
    #
    # in_diff_scores = np.array(in_diff_scores)
    #
    for key in ['imagenet', 'gaussian', 'uniform']:
        origin_out_result = np.load(RESULT_DIR + '/densenet_%s.npy' % key)

        out_results = []
        for i in range(L):
            out_result = np.load(RESULT_DIR + '/densenet_%s_%d.npy' % (key, i))
            # 目前版本值中已经softmax过，待改
            out_results.append(out_result)

        out_results = np.array(out_results)

        # 考虑全部变换，不找最大的一个了？？
        out_max_scores = np.max(origin_out_result, axis=1)
        # arr_stat('out max ', out_max_scores)
        # print(out_max_scores[:20])
        # exit(0)
        out_max_indices = np.argmax(origin_out_result, axis=1)
        out_sum_score = np.zeros(out_max_scores.shape)
        for i in range(L):
            # print(in_results[i].shape)
            # print(np.sum(in_results[i][0]))

            # 先算origin的，原始未经变换的结果

            # 对应的得分
            cor_scores = out_results[i][tuple(np.arange(out_results[i].shape[0])), tuple(out_max_indices)]
            # diff_score = cor_scores - out_max_scores
            diff_score = np.abs(cor_scores - out_max_scores)
            # print(key,'out ',diff_score[0:5])
            out_sum_score += diff_score

        # print('out sum', out_sum_score[:10])
        arr_stat('out sum', out_sum_score)
        evaluate_diff_score(key + ' sum ', -in_sum_score.reshape((in_sum_score.shape[0], 1)),
                            -out_sum_score.reshape((in_sum_score.shape[0], 1)))
    #         dsg0 = diff_score[diff_score > 0]
    #         print('out ',key,i)
    #         print(dsg0.shape)
    #         print(np.sum(dsg0))
    #         dsl0 = diff_score[diff_score < 0]
    #         print(dsl0.shape)
    #         print(np.sum(dsl0))

    #
    #
    # out_diff_scores = calculate_out_diff_scores(key, origin_out_result)
    #
    # for i in range(10):
    #     evaluate_diff_score('%s %d' % (key, i), in_diff_scores[i], out_diff_scores[i], True)
