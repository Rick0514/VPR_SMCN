import numpy as np
import matplotlib.pyplot as plt
import utils

# ---------------------------- 说明 ----------------------------------
# 证明环视数据集的优越性，通过对环视描述子压缩，并和原维度单视图描述子比较AP
# 每行打(*)号的需要根据实际情况配置
# 需要omni-SCUT数据集的描述子文件.npy
# ---------------------------- 说明 ----------------------------------

D1 = np.load('./experiments/cp4_1/day_desc.npy')    #(*)
D2 = np.load('./experiments/cp4_1/dawn_desc.npy')   #(*)

D1 = D1 / np.linalg.norm(D1, axis=0)
D2 = D2 / np.linalg.norm(D2, axis=0)

num = D1.shape[0]
omni_D1 = D1.reshape((num // 4, -1))
omni_D2 = D2.reshape((num // 4, -1))

GT = utils.makeGT(num // 4, 1)

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


P, R = utils.drawPR(S_pairwise, GT)
ap_pairwise = utils.calAvgPred(P, R)
P, R = utils.drawPR(S_slbsh, GT)
ap_slsbh = utils.calAvgPred(P, R)


# calulate the result and save them, firstly uncomment following code and
# run them, and comment it and visualize the result
# old_dims = omni_D1.shape[1]
# newr = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])  #(*)
# new_dims = (8192 * newr).astype(np.int)
# s = 0.25
#
# omni_ap = []
# for each_dim in new_dims:
#     P = np.random.rand(old_dims, each_dim)
#     P /= np.linalg.norm(P, axis=1, keepdims=True)
#
#     omni_D1_slsbh = utils.getLSBH(omni_D1, P, s)
#     omni_D2_slsbh = utils.getLSBH(omni_D2, P, s)
#
#     del P
#     nOnes = 2 * s * each_dim
#     S = np.matmul(omni_D1_slsbh, omni_D2_slsbh.T, dtype=np.float) / nOnes
#     P, R = utils.drawPR(S, GT)
#     ap = utils.calAvgPred(P, R)
#     omni_ap.append(ap)
#
# print(omni_ap)
# np.save('./experiments/cp4_1/omni_ap.npy', omni_ap)   #(*)

# visualize
omni_ap = np.load('./experiments/cp4_1/omni_ap.npy')    #(*)
newr = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
new_dims = 2 * (8192 * newr).astype(np.int)

fig = plt.figure()
ax1 = fig.add_subplot(111)
le1, = ax1.plot(new_dims, omni_ap, color=utils.color[0], marker='o', linewidth=2, markersize=10)
le2 = ax1.scatter([2 * 8192], [ap_pairwise], s=150, color=utils.color[1], marker='x')
le3 = ax1.scatter([2 * 8192], [ap_slsbh], s=150, color=utils.color[2], marker='p')

ax1.set_ylabel('AP')
ax1.set_ylim([0, 1])
# ax1.set_title("Double Y axis")
ax2 = ax1.twinx()  # this is the important function
le4, = ax2.plot(new_dims, newr, color=utils.color[3], marker='*', linewidth=2, markersize=10)
le5 = ax2.scatter([2 * 8192], [1.0], s=150, color=utils.color[2], marker='+')

ax2.set_ylabel('Memory')
ax1.set_xlabel('descriptor dimensions')
ax1.grid()
# ax2.text(x=1000,#文本x轴坐标
#          y=10, #文本y轴坐标
#          s='only cost 0.235s')
ax1.legend([le1, le2, le3, le4, le5], ['omni+sLSBH AP', 'sLSBH AP', 'pairwise AP',
            'omni-sLSBH memory consumption', 'sLSBH memory consumption'], loc=4)
plt.show()
