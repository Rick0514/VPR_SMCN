import numpy as np
import matplotlib.pyplot as plt
import main.utils as utils
import main.MCN as MCN
import main.SMCN as SMCN

# ---------------------------- 说明 ----------------------------------
# 将后端方法应用到环视数据集
# 每行打(*)号的需要根据实际情况配置
# 需要omni-SCUT数据集的描述子文件.npy
# ---------------------------- 说明 ----------------------------------

D1 = np.load('./experiments/cp4_1/netvlad/day_desc.npy')        #(*)
D2 = np.load('./experiments/cp4_1/netvlad/dawn_desc.npy')       #(*)

D1 = D1 / np.linalg.norm(D1, axis=0)
D2 = D2 / np.linalg.norm(D2, axis=0)

num = D1.shape[0]
omni_D1 = D1.reshape((num // 4, -1))
omni_D2 = D2.reshape((num // 4, -1))

GT = utils.getGroundTruthMatrix(num // 4, 1)

sig_D1 = D1[::4, :]
sig_D2 = D2[::4, :]

del D1, D2

# single view
num, old_dims = sig_D1.shape
new_dims = 8192     # acturally 8192 * 2
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)

sig_D1_slsbh = utils.getLSBH(sig_D1, P, 0.25)
sig_D2_slsbh = utils.getLSBH(sig_D2, P, 0.25)
del P

# pairwise
n1 = np.linalg.norm(sig_D1, axis=1).reshape((-1, 1))
n2 = np.linalg.norm(sig_D2, axis=1).reshape((1, -1))
n12 = np.matmul(n1, n2)
S_pairwise = np.matmul(sig_D1, sig_D2.T) / n12

# sLBSH
nOnes = 2 * s * new_dims
S_slbsh = np.matmul(sig_D1_slsbh, sig_D2_slsbh.T, dtype=np.float) / nOnes


# omni
old_dims = omni_D1.shape[1]
newr = 0.01     #(*) only use 1% memery of before
new_dims = int(8192 * newr)
s = 0.25

omni_ap = []

P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)

omni_D1_slsbh = utils.getLSBH(omni_D1, P, s)
omni_D2_slsbh = utils.getLSBH(omni_D2, P, s)

del P
nOnes = 2 * s * new_dims
S_omni = np.matmul(omni_D1_slsbh, omni_D2_slsbh.T, dtype=np.float) / nOnes
S_omni_in = np.matmul(omni_D1_slsbh, omni_D1_slsbh.T, dtype=np.float) / nOnes

# MCN
params = MCN.MCNParams(probAddCon=0.05,
                       nCellPerCol=32,
                       nConPerCol=200,
                       minColActivity=0.75, #(*) NetVLAD -> 0.65 AlexNet -> 0.75
                       nColPerPattern=50,
                       kActiveCol=100)

S_omni_mcn = MCN.runMCN_SDR(params, omni_D1_slsbh, omni_D2_slsbh, GT)

# SMCN
can = SMCN.getCandidate(S_omni_in, S_omni, *(2, 1, 3))
S_omni_smcncan = np.zeros_like(S_omni)
for i in range(can.shape[1]):
    tmp = np.where(can[:, i] >= 0)[0]
    if len(tmp):
        S_omni_smcncan[can[tmp, i], i] = S_omni[can[tmp, i], i]

# SMCNTF
S_omni_smcntf = SMCN.refreshLoss(S_omni_in, S_omni, can, 1)

# visulize
plt.figure()
P, R = utils.drawPR(S_pairwise, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[0], label='pairwise: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_slbsh, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[1], label='sLBSH: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_omni, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[2], label='omni-sLBSH%d : %.4f' % (new_dims * 2, ap), linewidth=2)

P, R = utils.drawPR(S_omni_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[3], label='omni-MCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_omni_smcncan, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[4], label='omni-SMCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_omni_smcntf, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=utils.color[5], label='omni-SMCNTF: %.4f' % ap, linewidth=2)


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('AlexNet')
plt.grid()

