import os
import random

import torch as t
import torch.nn.functional as F
import numpy as np


def innerProduct(usrEmbeds, itmEmbeds):
    return t.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def same_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    t.manual_seed(seed)  # 固定随机种子（CPU）
    if t.cuda.is_available():  # 固定随机种子（GPU)
        t.cuda.manual_seed(seed)  # 为当前GPU设置
        t.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    t.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    t.backends.cudnn.deterministic = True  # 固定网络结构


def contrast(nodes, allEmbeds, allEmbeds2=None):
    if allEmbeds2 is not None:
        pckEmbeds = allEmbeds[nodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
    else:
        uniqNodes = t.unique(nodes)
        pckEmbeds = allEmbeds[uniqNodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
    return scores


def contrastNCE(nodes, allEmbeds, allEmbeds2=None):
    if allEmbeds2 is not None:
        pckEmbeds = allEmbeds[nodes]
        pckEmbeds2 = allEmbeds2[nodes]
        # posScore = t.sum(pckEmbeds * pckEmbeds2)
        scores = t.log(t.exp(pckEmbeds * pckEmbeds2).sum(-1)).mean()
        # ssl_score = scores - posScore

    return scores


def calcReward(lastLosses, eps):
    if len(lastLosses) < 3:
        return 1.0
    curDecrease = lastLosses[-2] - lastLosses[-1]
    avgDecrease = 0
    for i in range(len(lastLosses) - 2):
        avgDecrease += lastLosses[i] - lastLosses[i + 1]
    avgDecrease /= len(lastLosses) - 2
    return 1 if curDecrease > avgDecrease else eps
