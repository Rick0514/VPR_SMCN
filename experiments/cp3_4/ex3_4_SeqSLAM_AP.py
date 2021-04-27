import numpy as np
import os
from scipy.io import loadmat
import utils

# ---------------------------- 说明 ----------------------------------
# 计算所有数据集上SeqSLAM的AP值
# 需要 1. 包含SeqSLAM输出的相似度矩阵的文件夹
# 其中文件夹分别为matlab生成的相似度矩阵文件：
# - gp_dldr.mat
# - gp_dlnr.mat
# - nd_sprwin.mat
# - ...
# 每行打(*)号的需要根据实际情况配置
# 输出各个相似矩阵的AP值
# ---------------------------- 说明 ----------------------------------

root = './experiments/cp3_4/seqslam'    #(*)

for each in os.listdir(root):
    S_file = os.path.join(root, each)
    S = loadmat(S_file)['S']
    tmp = np.isnan(S)
    S[tmp] = np.max(S[~tmp])

    num = S.shape[0]
    if each.startswith('nd'):
        GT = utils.makeGT(num, 9)   #(*) 9 is set in paper, which can be customised

    elif each.startswith('gp'):
        GT = utils.makeGT(num, 3)   #(*)

    elif each.startswith('ox'):
        r = '../../datasets/oxford_robotcar/'       #(*)
        if each.startswith('ox_daynight'):
            db_gps = np.load(r + 'gps_day.npy')     #(*) gps for the groundtruth
            q_gps = np.load(r + 'gps_night.npy')    #(*)
            _, GT = utils.getGpsGT(db_gps, q_gps, 5)#(*)
        elif each.startswith('ox_daysnow'):
            db_gps = np.load(r + 'gps_day.npy')     #(*)
            q_gps = np.load(r + 'gps_snow.npy')     #(*)
            _, GT = utils.getGpsGT(db_gps, q_gps, 5)#(*)
        else:
            db_gps = np.load(r + 'gps_snow.npy')    #(*)
            q_gps = np.load(r + 'gps_night.npy')    #(*)
            _, GT = utils.getGpsGT(db_gps, q_gps, 5)#(*)

    else:
        GT = utils.makeGT(num, 1)   #(*)

    P, R = utils.drawPR(S, GT, True)
    ap = utils.calAvgPred(P, R)
    print(each + " : %.4f" % ap)


