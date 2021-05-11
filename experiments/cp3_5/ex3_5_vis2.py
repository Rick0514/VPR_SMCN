import numpy as np
import main.utils as utils
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import os
from random import sample
import pickle

# ---------------------------- 说明 ----------------------------------
# 可视化图像检索效果图
# 需要 1. database和query数据集路径  2. xxx.npz文件（其中保存了相似度矩阵）
# 3. SMCNTF生成的最优路径文件 pathid.pkl
# 4. SeqSLAM生成的相似度矩阵 xxx.mat
# 每行打(*)号的需要根据实际情况配置
# 在论文中，对比了pairwise、MCN、SeqSLAM、SMCN和SMCNTF一共5种方法
# ---------------------------- 说明 ----------------------------------


# dataset root dir
root = '../../datasets/scut/'   #(*)
# load xxx.npz
S_file = './experiments/cp3_5/scut.npz'     #(*)
S = np.load(S_file)

S_pw = S['S_pw']
S_mcn = S['S_mcn']
# use SeqSLAM2.0 toolbox from matlab, the similarity matrix has much nan value
S_seq = loadmat('./experiments/cp3_4/seqslam/scut.mat')['S']    #(*)
tmp = np.isnan(S_seq)
S_seq[tmp] = np.max(S_seq[~tmp])
S_smcn = S['S_smcn']
S_smcntf = S['S_smcntf']


dbfile = 'day/'     #(*)    database file
qfile = 'dawn/'     #(*)    query file
dbn = len(os.listdir(root + dbfile))
qn = len(os.listdir(root + qfile))
qn = min(dbn, qn)

# if the dataset is SCUT, uncomment following code
numPicToShow = 10   #(*)
picid = sample(range(0, qn // 4), numPicToShow)
# otherwise, uncomment following code
# picid = sample(range(0, qn), numPicToShow)
numpic = len(picid)
img_format = '%d.jpg'   #(*)
err = 1                 #(*)
saveName = './experiments/cp3_5/vis2/scut.png'   #(*)

# load groundtruth
# if not the oxford robotcar dataset(or if not use gps as groundtruth)
# uncomment following code
gt = utils.makeGT(qn, err)

# otherwise uncomment following code
# r = '../../datasets/oxford_robotcar/'
# db_gps = np.load(r + 'gps_snow.npy')
# q_gps = np.load(r + 'gps_night.npy')
# _, gt = utils.getGpsGT(db_gps, q_gps, err)
gtl = []
for each in picid:
    gtl.append(list(np.where(gt[:, each])[0]))

id_pw = np.argmax(S_pw[:, picid], axis=0)
id_mcn = np.argmax(S_mcn[:, picid], axis=0)
id_smcn = np.argmax(S_smcn[:, picid], axis=0)
with open('./experiments/cp3_5/pathid.pkl', 'rb') as f:     #(*)
    id_smcntf = pickle.load(f)['scut']
id_smcntf = [id_smcntf[x] for x in picid]
id_seq = np.argmin(S_seq[:, picid], axis=0)

numMethods = 5      #(*)    how many methords to show

# 一下代码一般不需要配置
# --------------------------draw--------------------------------
pad = 7
img_size = 120
visImg = 255 * np.ones(((numMethods + 1)*(img_size+2*pad), numpic*(img_size + 2*pad), 3), dtype=np.uint8)

vboard = 0
hboard = 0
for i in range(numpic):
    img = cv2.imread(root + dbfile + img_format % picid[i])
    img = cv2.resize(img, (img_size, img_size))
    visImg[vboard+pad:vboard+pad+img_size,
            hboard+pad:hboard+pad+img_size :] = img
    hboard += (2*pad+img_size)
vboard += (2*pad + img_size)


hboard = 0
for i in range(numpic):
    img = cv2.imread(root + qfile + img_format % id_pw[i])
    img = cv2.resize(img, (img_size, img_size))
    if id_pw[i] in gtl[i]:
        visImg[vboard:vboard + 2*pad + img_size,
            hboard:hboard + 2*pad+img_size, :] = np.array([[[0, 255, 0]]])
    else:
        visImg[vboard:vboard + 2 * pad + img_size,
            hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 0, 255]]])

    visImg[vboard + pad:vboard + pad + img_size,
        hboard + pad:hboard + pad + img_size:] = img

    hboard += (2*pad+img_size)

vboard += (2*pad + img_size)

hboard = 0
for i in range(numpic):
    img = cv2.imread(root + qfile + img_format % id_seq[i])
    img = cv2.resize(img, (img_size, img_size))
    if id_seq[i] in gtl[i]:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 255, 0]]])
    else:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 0, 255]]])

    visImg[vboard + pad:vboard + pad + img_size,
    hboard + pad:hboard + pad + img_size:] = img

    hboard += (2*pad+img_size)

vboard += (2*pad + img_size)

hboard = 0
for i in range(numpic):
    img = cv2.imread(root + qfile + img_format % id_mcn[i])
    img = cv2.resize(img, (img_size, img_size))
    if id_mcn[i] in gtl[i]:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 255, 0]]])
    else:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 0, 255]]])

    visImg[vboard + pad:vboard + pad + img_size,
    hboard + pad:hboard + pad + img_size:] = img

    hboard += (2*pad+img_size)

vboard += (2*pad + img_size)

hboard = 0
for i in range(numpic):
    img = cv2.imread(root + qfile + img_format % id_smcn[i])
    img = cv2.resize(img, (img_size, img_size))
    if id_smcn[i] in gtl[i]:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 255, 0]]])
    else:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 0, 255]]])

    visImg[vboard + pad:vboard + pad + img_size,
    hboard + pad:hboard + pad + img_size:] = img

    hboard += (2*pad+img_size)

vboard += (2*pad + img_size)

hboard = 0
for i in range(numpic):
    img = cv2.imread(root + qfile + img_format % id_smcntf[i])
    img = cv2.resize(img, (img_size, img_size))
    if id_smcntf[i] in gtl[i]:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 255, 0]]])
    else:
        visImg[vboard:vboard + 2 * pad + img_size,
        hboard:hboard + 2 * pad + img_size, :] = np.array([[[0, 0, 255]]])

    visImg[vboard + pad:vboard + pad + img_size,
    hboard + pad:hboard + pad + img_size:] = img

    hboard += (2*pad+img_size)

vboard += (2*pad + img_size)

cv2.imshow('res', visImg)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite(saveName, visImg)