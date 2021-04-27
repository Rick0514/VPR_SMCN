import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from collections import namedtuple
import numpy as np
import utils
import time
from os.path import join, exists, isfile
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 把所有dataset下的子文件夹的图片转化为描述子，保存成.npy格式

yaml_path = './config.yaml'
cont = None
with open(yaml_path, 'r', encoding='utf-8') as f:
    cont = f.read()

arg = yaml.load(cont)
arg_dataset = arg['dataset']
arg_net = arg['net']

if arg_net['net_name'] == 'netvlad':
    net = utils.MyNetVLAD(ckpt=arg_net['resume_model'], nGPU=arg_net['nGPU'])
    net()

elif arg_net['net_name'] == 'alexnet':
    net = utils.MyAlexNet(ckpt=arg_net['resume_model'])
    net()

imgdir = join(arg_dataset['root'], arg_dataset['name'])
savedir = join(arg_net['save_desc'], arg_net['net_name'], arg_dataset['name'])
for each_dir in arg_dataset['subdir']:
    print('start to convert ' + each_dir + ' descriptors')
    dir = join(imgdir, each_dir)
    imgName = arg_dataset['postfix'] + arg_dataset['numformat'] + arg_dataset['suffix']
    rshape = (arg_dataset['resize']['w'], arg_dataset['resize']['h'])
    dataset = utils.MyDataset(dir, imgName, rshape)
    loader = data.DataLoader(dataset=dataset, num_workers=arg_net['threads'],
            batch_size=arg_net['batch_size'], shuffle=False, pin_memory=True)

    feat = net.genFeatures(loader, len(dataset))

    if arg_net['pca_flag']:
        feat = utils.pca_whitening(feat, arg_net['pca_dims'])
    
    save_path = join(savedir, each_dir + '_desc.npy')
    np.save(save_path, feat)
    
    print(each_dir + ' \'s descriptors are saved')

