from scipy.io import loadmat
import numpy as np
import main.utils as utils
import main.MCN as MCN
import main.SMCN as SMCN
import matplotlib.pyplot as plt
import time
import tools.visualize as vis

# with this code, we check our implementation with original code in matlab

# D1 = loadmat('../desc/summer_alexnet_lsh.mat')
# D2 = loadmat('../desc/winter_alexnet_lsh.mat')
#
# D1 = D1['D'] / np.linalg.norm(D1['D'], axis=1, keepdims=True)
# D2 = D2['D'] / np.linalg.norm(D2['D'], axis=1, keepdims=True)

D1 = np.load('../desc/netvlad/gp/day_right_desc.npy')
D2 = np.load('../desc/netvlad/gp/night_right_desc.npy')
D1 = D1 / np.linalg.norm(D1, axis=1, keepdims=True)
D2 = D2 / np.linalg.norm(D2, axis=1, keepdims=True)

num, old_dims = D1.shape
new_dims = 16384
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=0, keepdims=True)

D1_slsbh = utils.getLSBH(D1, P, 0.25)
D2_slsbh = utils.getLSBH(D2, P, 0.25)

del P

# =============== pair-wise ========================
n1 = np.linalg.norm(D1, axis=1).reshape((-1, 1))
n2 = np.linalg.norm(D2, axis=1).reshape((1, -1))
n12 = np.matmul(n1, n2)
S_pairwise = np.matmul(D1, D2.T) / n12
S_in = np.matmul(D1, D1.T) / np.matmul(n1, n1.T)
siid = np.argsort(S_in, axis=0)
# =============== pair-can ========================
# candidate
cannum = 5
spid = np.argsort(S_pairwise, axis=0)
S_can = np.zeros_like(S_pairwise)
for i in range(num):
    S_can[spid[-cannum:, i], i] = S_pairwise[spid[-cannum:, i], i]

del D1, D2

# =============== sLBSH ========================
nOnes = 2 * s * new_dims
S_slbsh = np.matmul(D1_slsbh, D2_slsbh.T, dtype=np.float) / nOnes

# =============== MCN ========================
st = time.time()
params = MCN.MCNParams(probAddCon=0.05,
                       nCellPerCol=32,
                       nConPerCol=200,
                       minColActivity=0.75,
                       nColPerPattern=50,
                       kActiveCol=100)
mcn = MCN.MCN(params)
train_winnerCells = []
for i in range(D1_slsbh.shape[0]):
    train_winnerCells.append(mcn.compute(D1_slsbh[i, :], False))

valid_winnerCells = []
mcn.resetPredP()
for i in range(D2_slsbh.shape[0]):
    valid_winnerCells.append(mcn.compute(D2_slsbh[i, :], True))

# get similarity matrix
S_mcn = np.zeros_like(S_slbsh)
for k1, each_v in enumerate(valid_winnerCells):
    for k2, each_t in enumerate(train_winnerCells):
        S_mcn[k2, k1] = MCN.getSim(each_v, each_t)
del train_winnerCells, valid_winnerCells, mcn
print("MCN: %.3f" % (time.time() - st))

# =============== SMCN ========================
# S_smcncan = np.zeros_like(S_pairwise)
# can = SMCN.getCandidate(S_in, S_pairwise, cannum, 2, 18)
# for i in range(num):
#     tmp = np.where(can[:, i] >= 0)[0]
#     if len(tmp):
#         S_smcncan[can[tmp, i], i] = S_pairwise[can[tmp, i], i]

# =============== SMCN-TF ========================
# S_smcn = smcn.SMCN_S(S_pairwise, S_in, can, wcan)
# del smcn, can, wcan
# err = 3
# S_smcntf = SMCN.refreshLoss(S_in, S_pairwise, can, err)
# ============================================

GT = utils.getGroundTruthMatrix(num, 1)

plt.figure()
P, R = utils.drawPR(S_pairwise, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[0], label='pairwise: %.4f' % ap)

# P, R = utils.drawPR(S_can, GT)
# ap = utils.calAvgPred(P, R)
# plt.plot(R, P, color=vis.color[1], label='paircan: %.4f' % ap)

P, R = utils.drawPR(S_slbsh, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[1], label='sLBSH: %.4f' % ap)

P, R = utils.drawPR(S_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[2], label='MCN: %.4f' % ap)

# P, R = utils.drawPR(S_smcncan, GT)
# ap = utils.calAvgPred(P, R)
# plt.plot(R, P, color=vis.color[4], label='SMCN-can: %.4f' % ap)
#
# P, R = utils.drawPR(S_smcntf, GT, True)
# ap = utils.calAvgPred(P, R)
# plt.plot(R, P, color=vis.color[5], label='SMCN-tf: %.4f' % ap)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid()