import time
import numpy as np
import utils

# ---------------------------- 说明 ----------------------------------
# SMCN和SMCNTF算法
# 1. 先运行getCandidate获得使用SMCN生成的候选矩阵
# 2. 然后运行refreshLoss会的候选矩阵的候选相似度矩阵，即SMCNTF
# ---------------------------- 说明 ----------------------------------


def getCandidate(Si, Sp, cannum, k1, k2):

    topk_i = k1
    topk_p = k2
    dbnum = Si.shape[1]
    # train
    Si_id = np.argsort(-Si, axis=0)[:topk_i, :]
    preConnections = [set()] * dbnum
    for i in range(dbnum-1):
        for j in Si_id[:, i]:
            preConnections[j] = set.union(preConnections[j], Si_id[:, i+1])

    # inference
    qnum = Sp.shape[1]
    Sp_id = np.argsort(-Sp, axis=0)[:topk_p, :]
    can = - np.ones((cannum, qnum), dtype=np.int)
    can[:, 0] = Sp_id[:cannum, 0]
    prev = can[:, 0]
    for i in range(1, qnum):
        pcan = Sp_id[:, i]
        pred = np.empty((0, ), dtype=np.int)
        for j in prev:
            pred = np.concatenate((pred, list(preConnections[j])))
        pcan = pcan[np.in1d(pcan, pred)]
        tmpl = min(cannum, len(pcan))
        if tmpl:
            can[:tmpl, i] = pcan[:tmpl]
            prev = pcan
        else:
            can[:, i] = Sp_id[:cannum, i]
            prev = can[:, i]

    return can

def refreshLoss(S_in, S, candidate, err):

    cannum, num = candidate.shape
    loss = - np.ones_like(candidate, dtype=np.float)
    lossid = - np.ones_like(loss, dtype=np.int)
    for i in range(num):
        tmp = np.where(candidate[:, i] >= 0)[0]
        loss[tmp, i] = 1 - S[tmp, i]

    # dynamic programming
    lastid = np.where(candidate[:, 0] >= 0)[0]
    for i in range(1, num):
        canid = np.where(candidate[:, i] >= 0)[0]
        for each_canid in canid:
            lossterm = np.zeros_like(lastid, dtype=np.float)
            lossterm += loss[lastid, i-1]
            lossterm += loss[each_canid, i]
            for k, eachid in enumerate(lastid):
                if abs(candidate[each_canid, i] - candidate[eachid, i-1]) > err:
                    lossterm[k] += (1 - S_in[candidate[each_canid, i], candidate[eachid, i-1]])

            tmpid = np.argmin(lossterm)
            lossid[each_canid, i] = tmpid
            loss[each_canid, i] = lossterm[tmpid]
        lastid = canid

    # find the most continuous path
    tmpid = np.where(loss[:, -1] >= 0)[0]
    minid = np.argmin(loss[tmpid, -1])
    pathid = [candidate[minid, -1]]
    for i in range(num-1, 0, -1):
        minid = lossid[minid, i]
        pathid.append(candidate[minid, i-1])
    pathid.reverse()

    rfS = np.ones_like(S)
    for i in range(num):
        tmp = np.where(candidate[:, i] >= 0)[0]
        canid = candidate[tmp, i]
        for each_canid in canid:
            if abs(each_canid - pathid[i]) > err:
                rfS[each_canid, i] = 1 - S_in[each_canid, pathid[i]]
            else:
                rfS[each_canid, i] = 0

    return rfS, pathid


def runSMCN(dbFeat, qFeat, gt, SMCN_params, flag):
    st = time.time()
    n1 = np.linalg.norm(dbFeat, axis=1).reshape((-1, 1))
    n2 = np.linalg.norm(qFeat, axis=1).reshape((1, -1))
    n12 = np.matmul(n1, n2)
    S_pairwise = np.matmul(dbFeat, qFeat.T) / n12
    S_in = np.matmul(dbFeat, dbFeat.T) / np.matmul(n1, n1.T)

    can = getCandidate(S_in, S_pairwise, *SMCN_params)
    S_smcncan = np.zeros_like(S_pairwise)
    for i in range(can.shape[1]):
        tmp = np.where(can[:, i] >= 0)[0]
        if len(tmp):
            S_smcncan[can[tmp, i], i] = S_pairwise[can[tmp, i], i]

    if flag == 's':
        return S_smcncan
    else:
        time_cost = time.time() - st
        P, R = utils.drawPR(S_smcncan, gt)
        ap = utils.calAvgPred(P, R)
        return ap, time_cost


def runSMCNTF(dbFeat, qFeat, gt, SMCN_params, err, flag):

    st = time.time()
    n1 = np.linalg.norm(dbFeat, axis=1).reshape((-1, 1))
    n2 = np.linalg.norm(qFeat, axis=1).reshape((1, -1))
    n12 = np.matmul(n1, n2)
    S_pairwise = np.matmul(dbFeat, qFeat.T) / n12
    S_in = np.matmul(dbFeat, dbFeat.T) / np.matmul(n1, n1.T)

    can = getCandidate(S_in, S_pairwise, *SMCN_params)
    S_smcntf, pathid = refreshLoss(S_in, S_pairwise, can, err)
    time_cost = time.time() - st

    if flag == 's':
        return S_smcntf
    elif flag == 'r':
        return pathid
    else:
        P, R = utils.drawPR(S_smcntf, gt, True)
        ap = utils.calAvgPred(P, R)
        return ap, time_cost
