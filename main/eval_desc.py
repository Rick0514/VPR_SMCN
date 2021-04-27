import numpy as np
import yaml
import pickle
from os import listdir
from os.path import join
import faiss
import utils

# load global config yaml
yaml_path = './config.yaml'
cont = None
with open(yaml_path, 'r', encoding='utf-8') as f:
    cont = f.read()

arg = yaml.load(cont)
arg_dataset = arg['dataset']
arg_net = arg['net']
arg_eval = arg['evaluate']


desc_dir = join(arg_net['save_desc'], arg_net['net_name'], arg_dataset['name'])
# load features
subdir = arg_dataset['subdir']
db_save_path = join(desc_dir, subdir[arg_eval['compare_subdir'][0]] + '_desc.npy')
q_save_path = join(desc_dir, subdir[arg_eval['compare_subdir'][1]] + '_desc.npy')

dbFeat = np.load(db_save_path)
qFeat = np.load(q_save_path)


num, dim = dbFeat.shape

# get ground truth
err = 3
gt = utils.getGroundTruth(num, err)
gtm = utils.getGroundTruthMatrix(num, err)

_ = utils.getResult(dbFeat, qFeat, gt, gtm)

S = utils.MCN_pairwise(dbFeat, qFeat)
P, R = utils.drawPR(S, gtm)
auc = utils.calAvgPred(P, R)

print("auc : %.4f" % auc)

# test dimension reduction effect
# r_dim = 128
# w_dbFeat = utils.pca_whitening(dbFeat, r_dim)
# w_dbFeat = np.ascontiguousarray(w_dbFeat)
# w_qFeat = utils.pca_whitening(qFeat, r_dim)
# w_qFeat = np.ascontiguousarray(w_qFeat)
# w_S = utils.getResult(w_dbFeat, w_qFeat, gt, gtm)

# # save S
# res_dir = join(arg_eval['save_res'], arg_net['net_name'], arg_dataset['name'])
# save_path = join(res_dir,
#     subdir[arg_eval['compare_subdir'][0]] + '_' 
#     + subdir[arg_eval['compare_subdir'][1]] + '.npy')

# np.save(save_path, S)


     

