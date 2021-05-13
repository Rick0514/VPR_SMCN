import numpy as np
import main.utils as utils
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import tools.visualize as vis

# ---------------------------- 说明 ----------------------------------
# 可视化在不同数据集上各种方法的PR曲线
# 需要 1. 相似度矩阵文件xxx.npz
# 每行打(*)号的需要根据实际情况配置
# 在论文中，使用netvlad描述子
# 生成pw，pwk，MCN，SeqSLAM，SMCN，SMCNTF六种方法
# 在NL(summer vs. winter), GP(day_left vs. day_right)
# OX(snow vs. night), SCUT(day vs. dawn)
# 一共4张图
# ---------------------------- 说明 ----------------------------------

with open('../cp3_4/s_dd.pkl', 'rb') as f:
    S_gp_dd = pickle.load(f)
    S_nd_dd = pickle.load(f)
    S_ox_dd = pickle.load(f)
    S_sc_dd = pickle.load(f)

# ---------------------------- draw NL --------------------------
plt.figure()
err = 9     #(*) error tolerance for nordland dataset, 9 is set in paper
S_file = '../cp3_5/vis3/nd_sumwin.npz'    #(*)    NL similarity matrix
S = np.load(S_file)
S_pw = S['S_pw']
S_pwk = S['S_pwk']
S_mcn = S['S_mcn']
S_seq = loadmat('../cp3_4/seqslam/nd_sumwin.mat')['S']   #(*)
tmp = np.isnan(S_seq)
S_seq[tmp] = np.max(S_seq[~tmp])
S_smcn = S['S_smcn']
S_smcntf = S['S_smcntf']
S_dd = S_nd_dd
del S

GT = utils.getGroundTruthMatrix(S_pw.shape[0], err)
P, R = utils.drawPR(S_pw, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[0], label='PW: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_pwk, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[1], label='PWk: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[2], label='MCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_seq, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[3], label='SeqSLAM: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_dd, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[4], label='Delta: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[5], label='SMCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcntf, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[6], label='SMCNTF: %.4f' % ap, linewidth=2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Nordland\nSummer vs. Winter')
plt.grid()

#----------------------------- draw GP -------------------------
err = 3     #(*)
plt.figure()
S_file = '../cp3_5/vis3/gp_dlnr.npz'  #(*)
S = np.load(S_file)
S_pw = S['S_pw']
S_pwk = S['S_pwk']
S_mcn = S['S_mcn']
S_seq = loadmat('../cp3_4/seqslam/gp_dlnr.mat')['S'] #(*)
tmp = np.isnan(S_seq)
S_seq[tmp] = np.max(S_seq[~tmp])
S_smcn = S['S_smcn']
S_smcntf = S['S_smcntf']
S_dd = S_gp_dd
del S

GT = utils.getGroundTruthMatrix(S_pw.shape[0], err)
P, R = utils.drawPR(S_pw, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[0], label='PW: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_pwk, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[1], label='PWk: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[2], label='MCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_seq, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[3], label='SeqSLAM: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_dd, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[4], label='Delta: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[5], label='SMCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcntf, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[6], label='SMCNTF: %.4f' % ap, linewidth=2)


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Gardens Point\nday_left vs. night_right')
plt.grid()

# ------------------- draw Oxford ----------------------------
plt.figure()
S_file = '../cp3_5/vis3/ox_sn.npz'    #(*)
S = np.load(S_file)
S_pw = S['S_pw']
S_pwk = S['S_pwk']
S_mcn = S['S_mcn']
S_seq = loadmat('../cp3_4/seqslam/ox_snownight.mat')['S']    #(*)
tmp = np.isnan(S_seq)
S_seq[tmp] = np.max(S_seq[~tmp])
S_smcn = S['S_smcn']
S_smcntf = S['S_smcntf']
S_dd = S_ox_dd
del S

# because gps groundthuth is used for oxford robotcar dataset
# therefore the gps file should be loaded
err = 5     #(*) which means 5 meters error tolerance, not the number of frames
r = 'E:/project/scut/graduation/datasets/oxford_robotcar/'   #(*)
db_gps = np.load(r + 'gps_snow.npy')    #(*)
q_gps = np.load(r + 'gps_night.npy')    #(*)
_, GT = utils.getGpsGT(db_gps, q_gps, err)

P, R = utils.drawPR(S_pw, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[0], label='PW: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_pwk, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[1], label='PWk: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[2], label='MCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_seq, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[3], label='SeqSLAM: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_dd, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[4], label='Delta: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[5], label='SMCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcntf, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[6], label='SMCNTF: %.4f' % ap, linewidth=2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Oxford Robotcar\nSnow vs. Night')
plt.grid()

# -------------------------- draw SCUT -------------------
err = 1     #(*)
plt.figure()
S_file = '../cp3_5/vis3/scut.npz' #(*)
S = np.load(S_file)
S_pw = S['S_pw']
S_pwk = S['S_pwk']
S_mcn = S['S_mcn']
S_seq = loadmat('../cp3_4/seqslam/scut.mat')['S']    #(*)
tmp = np.isnan(S_seq)
S_seq[tmp] = np.max(S_seq[~tmp])
S_smcn = S['S_smcn']
S_smcntf = S['S_smcntf']
S_dd = S_sc_dd
del S

GT = utils.getGroundTruthMatrix(S_pw.shape[0], err)
P, R = utils.drawPR(S_pw, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[0], label='PW: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_pwk, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[1], label='PWk: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_mcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[2], label='MCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_seq, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[3], label='SeqSLAM: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_dd, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[4], label='Delta: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcn, GT)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[5], label='SMCN: %.4f' % ap, linewidth=2)

P, R = utils.drawPR(S_smcntf, GT, True)
ap = utils.calAvgPred(P, R)
plt.plot(R, P, color=vis.color[6], label='SMCNTF: %.4f' % ap, linewidth=2)


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc=3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('SCUT\nDay vs. Dawn')
plt.grid()
