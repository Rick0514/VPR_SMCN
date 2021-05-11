import numpy as np
from tqdm import tqdm
import yaml
from os.path import join
import main.utils as utils


# --------------info-------------------
# we refer to https://github.com/oravus/DeltaDescriptors and
# modify its code to apply to our experiment
# --------------info-------------------

def performDelta(data, winL=None):
    v = (-1.0 * np.ones(winL)) / (winL / 2.0)
    v[:winL // 2] *= -1

    ftAll = []
    for ft in data:
        ftC = []  # ft1.copy(), ft2.copy()
        # note that np.convolve flips the v vector hence sign is inverted above
        for i1 in tqdm(range(ft.shape[1])):
            ftC.append(np.convolve(ft[:, i1], v, "same"))
        ftC = np.array(ftC).transpose()

        # a forced fix for zero-sum delta descs -> use the raw descriptor (could otherwise skip frames)
        ftC = np.array([ft[j] if f.sum() == 0 else f for j, f in enumerate(ftC)])
        ftAll.append(ftC)

    return ftAll


# load global config yaml
yaml_path = './config.yaml'  # (*)
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

dbFeat = np.load(db_save_path)
qFeat = np.load(q_save_path)
if arg_dataset['name'] == 'scut':
    dbFeat = dbFeat[::4, :]
    qFeat = qFeat[::4, :]

dbn = dbFeat.shape[0]
qn = qFeat.shape[0]

# get ground truth
if arg_dataset['name'] == 'ox_robotcar':
    err = arg_dataset['err']
    db_gps = np.load(join(imgdir, 'gps_' + subdir[arg_eval['compare_subdir'][0]] + '.npy'))
    q_gps = np.load(join(imgdir, 'gps_' + subdir[arg_eval['compare_subdir'][1]] + '.npy'))
    _, gtm = utils.getGpsGT(db_gps, q_gps, err)
else:
    err = arg_dataset['err']
    gtm = utils.getGroundTruthMatrix(dbn, err)

print("Computing Descriptors...")
seqLen = 16
descData = performDelta([dbFeat, qFeat], seqLen)

S = utils.MCN_pairwise(descData[0], descData[1])
P, R = utils.drawPR(S, gtm)
auc = utils.calAvgPred(P, R)
print("delta auc : %.4f" % auc)