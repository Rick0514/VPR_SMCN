import numpy as np
import yaml
import pickle
from os import listdir
from os.path import join
import faiss
import utils
import SMCN

# ---------------------------- 说明 ----------------------------------
# 候选矩阵有效性且给出候选个数的选择
# 需要 1. 配置文件 config.ymal
# 每行打(*)号的需要根据实际情况配置
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

S = utils.MCN_pairwise(dbFeat, qFeat)

# experiment 3_2
# if SCUT, uncomment following
# canr = [1, 3, 5, 7]
# otherwise
canr = [0.01, 0.02, 0.05, 0.1]      #(*)

spid = np.argsort(S, axis=0)
for each in canr:
    # if not SCUT                   #(*)
    cannum = int(each * dbn)
    # otherwise
    # cannum = each
    S_can = np.zeros_like(S)
    for i in range(qn):
        S_can[spid[-cannum:, i], i] = S[spid[-cannum:, i], i]
    
    P, R = utils.drawPR(S_can, gtm)
    auc = utils.calAvgPred(P, R)
    print('cannum: %d auc: %.4f' % (cannum, auc))
    
