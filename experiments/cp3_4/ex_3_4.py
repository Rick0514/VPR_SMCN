import numpy as np
import yaml
import pickle
from os import listdir
from os.path import join
import faiss
import utils
import SMCN, MCN
import pickle

# ---------------------------- 说明 ----------------------------------
# 计算所有数据集上在PW、PWK、MCN、SMCN和SMCNTF的AP值
# 需要 1. 配置文件 config.ymal
# 每行打(*)号的需要根据实际情况配置
# 输出各个相似矩阵的AP值
# ---------------------------- 说明 ----------------------------------


# load global config yaml
yaml_path = './config.yaml'     #(*)
cont = None
with open(yaml_path, 'r', encoding='utf-8') as f:
    cont = f.read()

arg = yaml.load(cont)
arg_dataset = arg['dataset']
arg_net = arg['net']
arg_eval = arg['evaluate']

imgdir = join(arg_dataset['root'], arg_dataset['name'])
desc_dir = join(arg_net['save_desc'], arg_net['net_name'], arg_dataset['name'])
# load features
subdir = arg_dataset['subdir']
db_save_path = join(desc_dir, subdir[arg_eval['compare_subdir'][0]] + '_desc.npy')
q_save_path = join(desc_dir, subdir[arg_eval['compare_subdir'][1]] + '_desc.npy')

# if use SCUT datasets, uncomment following code
dbFeat = np.load(db_save_path)[::4, :]  #(*)
qFeat = np.load(q_save_path)[::4, :]
# otherwise uncomment following code
# dbFeat = np.load(db_save_path)
# qFeat = np.load(q_save_path)


dbn = dbFeat.shape[0]
qn = qFeat.shape[0]

# get ground truth      #(*)
# if not use oxford robotcar, uncomment following code
err = arg_dataset['err']
gt = utils.getGroundTruth(dbn, err)
gtm = utils.getGroundTruthMatrix(dbn, err)
# oxford robotcar
# err = arg_dataset['err']
# db_gps = np.load(join(imgdir, 'gps_' + subdir[arg_eval['compare_subdir'][0]] + '.npy'))
# q_gps = np.load(join(imgdir, 'gps_' + subdir[arg_eval['compare_subdir'][1]] + '.npy'))
# _, gtm = utils.getGpsGT(db_gps, q_gps, err)

# following code is optional, if you want to see the recall@N
# you can uncomment it
# _ = utils.getResult(dbFeat, qFeat, gt, gtm)


# ------------- cal S pairwise ap -------------------
S_pw = utils.MCN_pairwise(dbFeat, qFeat)
P, R = utils.drawPR(S_pw, gtm)
auc = utils.calAvgPred(P, R)
print("S_pw auc : %.4f" % auc)

D1 = dbFeat / np.linalg.norm(dbFeat, axis=0)
D2 = qFeat / np.linalg.norm(qFeat, axis=0)
del dbFeat, qFeat

# if not SCUT
# cannum = int(0.02 * dbn)

# if oxford robotcar
# k1 = 2
# k2 = int(0.05 * dbn)

# if SCUT
k1 = 2
k2 = 4
cannum = 3

# # otherwise
# k1 = 4
# k2 = int(0.03 * dbn)

# ----------------- S_pwk ---------------------
spid = np.argsort(S_pw, axis=0)
S_pwk = np.zeros_like(S_pw)
for i in range(qn):
    S_pwk[spid[-cannum:, i], i] = S_pw[spid[-cannum:, i], i]

P, R = utils.drawPR(S_pwk, gtm)
ap = utils.calAvgPred(P, R)
print("S_pwk --> ap : %.4f" % ap)

# ---------------SMCN and SMCNTF-----------------------

ap1, t1 = SMCN.runSMCN(D1, D2, gtm, (cannum, k1, k2), err)
print("SMCN --> ap : %.4f, t : %.6f" % (ap1, t1))

_, pathid = SMCN.runSMCNTF(D1, D2, gtm, (cannum, k1, k2), err)
with open('./pathid.pkl', 'ab') as f:   #(*)    save the best match sequence
    pickle.dump(pathid, f)

print("SMCNTF --> ap : %.4f, t : %.6f" % (ap2, t2))


# -----------------MCN--------------------------
# in paper if NetVLAD minColActivity=0.65 
# if AlexNet minColActivity=0.75
params = MCN.MCNParams(probAddCon=0.05,     #(*)
                       nCellPerCol=32,
                       nConPerCol=200,
                       minColActivity=0.65,
                       nColPerPattern=50,
                       kActiveCol=100)

ap, t = MCN.runMCN(params, D1, D2, gtm)
print("MCN --> ap : %.4f, t : %.3f" % (ap, t))

#(*)    save most of the similarity matrix
np.savez('./nd_sumwin.npz', S_pw=S_pw, S_pwk=S_pwk, S_mcn=S_mcn, S_smcn=S_smcn, S_smcntf=S_smcntf)

