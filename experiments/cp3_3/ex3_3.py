import numpy as np
import yaml
import pickle
from os import listdir
from os.path import join
import main.utils as utils
import main.SMCN as SMCN

# ---------------------------- 说明 ----------------------------------
# 遍历SMCN的超参数
# 需要 1. 配置文件 config.ymal
# 每行打(*)号的需要根据实际情况配置
# 输出每组超参数得到的AP 保存在 .npy文件
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


# experiment 3_3
D1 = dbFeat / np.linalg.norm(dbFeat, axis=0)
D2 = qFeat / np.linalg.norm(qFeat, axis=0)
del dbFeat, qFeat
cannum = int(0.02 * D1.shape[0])
k1 = np.array([2, 3, 4, 5], dtype=np.int)
k2 = np.array([0.03, 0.05, 0.07, 0.09])
k2 *= D1.shape[0]
k2 = k2.astype(np.int)
res = np.zeros((4, 4))
for i, each_k1 in enumerate(k1):
    for j, each_k2 in enumerate(k2):
        ap, _ = SMCN.runSMCN(D1, D2, gtm, (cannum, each_k1, each_k2), err)
        res[i, j] = ap

print(res)
np.save(arg_net['net_name'] + '_' + arg_dataset['name'] + '.npy', res)  #(*)

