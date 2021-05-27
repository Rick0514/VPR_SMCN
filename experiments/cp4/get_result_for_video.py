import numpy as np
import matplotlib.pyplot as plt
import main.utils as utils
import tools.visualize as vis
import main.MCN as MCN
import main.SMCN as SMCN

# ---------------------------- 说明 ----------------------------------
# generate some result( especially simularity matrix )
# (*) means you should specify according to your case
# omni-SCUT descriptor file *.npy are needed
# ---------------------------- 说明 ----------------------------------

D1 = np.load('./netvlad/day_desc.npy')    #(*)
D2 = np.load('./netvlad/dawn_desc.npy')   #(*)

D1 = D1 / np.linalg.norm(D1, axis=0)
D2 = D2 / np.linalg.norm(D2, axis=0)

num = D1.shape[0]
omni_D1 = D1.reshape((num // 4, -1))
omni_D2 = D2.reshape((num // 4, -1))

GT = utils.getGroundTruthMatrix(num // 4, 1)

sig_D1 = D1[::4, :]
sig_D2 = D2[::4, :]

del D1, D2

id_dict = {}

# single view
num, old_dims = sig_D1.shape
new_dims = 8192     # acturally 8192 * 2
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)
sig_D1_slsbh = utils.getLSBH(sig_D1, P, 0.25)
sig_D2_slsbh = utils.getLSBH(sig_D2, P, 0.25)
del P
nOnes = 2 * s * new_dims
S_slbsh = np.matmul(sig_D1_slsbh, sig_D2_slsbh.T, dtype=np.float) / nOnes
id_dict['sig_f_id'] = np.argmax(S_slbsh, axis=0)
del S_slbsh, sig_D1_slsbh, sig_D2_slsbh

# onmi full
old_dims = omni_D1.shape[1]
new_dims = 8192
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)
omni_D1_slsbh = utils.getLSBH(omni_D1, P, s)
omni_D2_slsbh = utils.getLSBH(omni_D2, P, s)
del P
nOnes = 2 * s * new_dims
S = np.matmul(omni_D1_slsbh, omni_D2_slsbh.T, dtype=np.float) / nOnes
id_dict['onmi_f_id'] = np.argmax(S, axis=0)
del S, omni_D1_slsbh, omni_D2_slsbh

# 1% tiny single
_, old_dims = sig_D1.shape
new_dims = int(8192 * 0.01)     # acturally 8192 * 2
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)
sig_D1_slsbh = utils.getLSBH(sig_D1, P, 0.25)
sig_D2_slsbh = utils.getLSBH(sig_D2, P, 0.25)
del P
nOnes = 2 * s * new_dims
S_slbsh = np.matmul(sig_D1_slsbh, sig_D2_slsbh.T, dtype=np.float) / nOnes
id_dict['sig_t_id'] = np.argmax(S_slbsh, axis=0)
del S_slbsh, sig_D1_slsbh, sig_D2_slsbh

# 1% tiny omni
old_dims = omni_D1.shape[1]
new_dims = int(8192 * 0.01)
s = 0.25
P = np.random.rand(old_dims, new_dims)
P /= np.linalg.norm(P, axis=1, keepdims=True)
tiny_omni_D1_slsbh = utils.getLSBH(omni_D1, P, s)
tiny_omni_D2_slsbh = utils.getLSBH(omni_D2, P, s)
del P
nOnes = 2 * s * new_dims
S_pw = np.matmul(tiny_omni_D1_slsbh, tiny_omni_D2_slsbh.T, dtype=np.float) / nOnes
S_in = np.matmul(tiny_omni_D1_slsbh, tiny_omni_D2_slsbh.T, dtype=np.float) / nOnes
id_dict['onmi_t_id'] = np.argmax(S_pw, axis=0)

# 1% tiny omni MCN
params = MCN.MCNParams(probAddCon=0.05,     #(*)
                       nCellPerCol=32,
                       nConPerCol=200,
                       minColActivity=0.65,
                       nColPerPattern=50,
                       kActiveCol=100)

S = MCN.runMCN_SDR(params, tiny_omni_D1_slsbh, tiny_omni_D2_slsbh, GT)
id_dict['onmi_t_mcn_id'] = np.argmax(S, axis=0)

# 1% tiny omni SMCN
k1 = 1
k2 = 3
cannum = 2
can = SMCN.getCandidate(S_in, S_pw, *(cannum, k1, k2))
S_omni_smcncan = np.zeros_like(S_pw)
for i in range(can.shape[1]):
    tmp = np.where(can[:, i] >= 0)[0]
    if len(tmp):
        S_omni_smcncan[can[tmp, i], i] = S_pw[can[tmp, i], i]

id_dict['onmi_t_smcn_id'] = np.argmax(S_omni_smcncan, axis=0)

# 1% tiny omni SMCNTF
_, p = SMCN.refreshLoss(S_in, S_pw, can, 1)
id_dict['onmi_t_smcntf_id'] = p