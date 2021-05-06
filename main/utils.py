# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.utils.data as data
# import faiss
# from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import PCA
# import main.netvlad as netvlad

from os import _exit
from os.path import join, exists
from os import listdir

import numpy as np
from PIL import Image
import math
import pickle

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getLSBH(x, P, s):
    """
    x --> m x n with n features
    P --> random projection matrix: n x n1
    s --> sparsify rate
    return x --> m x n1
    """
    od, nd = P.shape
    n = round(nd * s)

    y = np.matmul(x, P)
    yid = np.argsort(y, axis=1)

    y1 = np.zeros_like(y, dtype=np.bool)
    y2 = np.zeros_like(y, dtype=np.bool)

    for i in range(x.shape[0]):
        y1[i, yid[i, :n]] = 1
        y2[i, yid[i, -n:]] = 1

    return np.concatenate((y1, y2), axis=1)


def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def getGpsGT(dbgps, qgps, err):
    qn = qgps.shape[1]
    dbn = dbgps.shape[1]
    dist = np.zeros((dbn, qn))
    for i in range(dbn):
        for j in range(qn):
            dist[i, j] = haversine(dbgps[:, i], qgps[:, j])

    gtm = (dist <= err)
    gt = []
    for i in range(qn):
        gt.append(np.where(gtm[:, i] == 1)[0])

    return gt, gtm

# def input_transform():
#     return transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225]),
#     ])
#
#
#
# class MyDataset(data.Dataset):
#     def __init__(self, file, imgName, rshape, input_transform=input_transform()):
#         super().__init__()
#
#         if exists(file):
#             self.images_path = file
#         else:
#             print("file wrong")
#             _exit(0)
#
#         self.input_transform = input_transform
#         self.rshape = rshape
#         self.imgName = imgName
#
#     def __getitem__(self, index):
#         # index start from 0
#         path = join(self.images_path, self.imgName % index)
#         if exists(path):
#             img = Image.open(path)
#             img = img.resize(self.rshape)
#
#             if self.input_transform:
#                 img = self.input_transform(img)
#         else:
#             print("path wrong")
#             _exit(0)
#
#         return img, index
#
#     def __len__(self):
#         return len(listdir(self.images_path))
#
#
# class OxDataset(data.Dataset):
#     def __init__(self, imgdir, gnd, rshape, input_transform=input_transform()):
#         super().__init__()
#
#         if exists(imgdir):
#             self.imgdir = imgdir
#         else:
#             print(imgdir + ' not found')
#             _exit(0)
#
#         if exists(gnd):
#             with open(gnd, 'rb') as f:
#                 cfg = pickle.load(f)
#
#             self.imlist = cfg['imlist'].extend(cfg['qimlist'])
#         else:
#             print(gnd + " not found")
#             _exit(0)
#
#         self.input_transform = input_transform
#         self.rshape = rshape
#
#     def __getitem__(self, index):
#         # index start from 0
#         path = join(self.imgdir, self.imlist[index])
#         if exists(path):
#             img = Image.open(path)
#             img = img.resize(self.rshape)
#
#             if self.input_transform:
#                 img = self.input_transform(img)
#         else:
#             print("path wrong")
#             _exit(0)
#
#         return img, index
#
#     def __len__(self):
#         return len(self.imlist)
#


def getGroundTruth(num, err):
    # q x db
    # err should be odd
    gt = -1 * np.ones((num, err))
    for i in range(num):
        st = max(0, i - err // 2)
        ed = min(num-1, i + err // 2)
        n = ed - st + 1
        gt[i, :n] = np.arange(st, ed+1, 1)
    
    return gt

def getGroundTruthMatrix(num, err):
    # err should be odd
    gt = np.eye(num, num, dtype=np.bool)
    for i in range(1, err):
        gt += np.eye(num, num, i, dtype=np.bool)
        gt += np.eye(num, num, -i, dtype=np.bool)
        
    return gt


def MCN_pairwise(dbFeat, qFeat):
    
    # normalise row
    dbFeat /= np.linalg.norm(dbFeat, axis=0)
    qFeat /= np.linalg.norm(qFeat, axis=0)

    dbNorm = np.linalg.norm(dbFeat, axis=1).reshape((-1, 1))
    qNorm = np.linalg.norm(qFeat, axis=1).reshape((1, -1))

    # cos
    S = np.matmul(dbFeat, qFeat.T) / np.matmul(dbNorm, qNorm)

    return S
        
def drawPR(S, GT, flag=False):
    """
    draw PR curve
    :param S: simularity matrix
    :param GT: grouthtruth matrix, should be logical
    :return: list P and R
    """

    R = [0]
    P = [1]
    if flag:
        th = np.linspace(np.min(S), np.max(S), 1000)
        sign = lambda x, y : x <= y
    else:
        th = np.linspace(np.max(S), np.min(S), 1000)
        sign = lambda x, y : x >= y
        
    for each_th in th:
        B = sign(S, each_th)
        TP = np.sum(np.logical_and(B, GT))
        FN = np.sum(np.logical_and(~B, GT))
        FP = np.sum(np.logical_and(B, ~GT))

        R.append(TP / (TP + FN))
        P.append(TP / (TP + FP))

    return P, R

def calAvgPred(P, R):
    """
    :param P: list type
    :param R:
    :param flag: False --> max True --> min
    :return: average precision, which is integral of PR curve
    """
    Pr = np.array(P)
    Re = np.array(R)
    inter = Re[1:] - Re[:-1]
    y1 = Pr[1:]
    y2 = Pr[:-1]

  
    return np.sum((y1 + y2) * inter) / 2


# def getResult(dbFeat, qFeat, gt, gtm):
#     # TODO what if features dont fit in memory?
#     # dbFeat --> num x dim
#     # gt --> num x dim
#     print('====> Building faiss index')
#     faiss_index = faiss.IndexFlatL2(dbFeat.shape[1])
#     faiss_index.add(dbFeat)
#
#     print('====> Calculating recall @ N')
#     n_values = [1,5,10]
#     n = dbFeat.shape[0]
#     all_dist, all_pred = faiss_index.search(qFeat, n)
#     predictions = all_pred[:, :n_values[-1]]
#     dist = np.zeros_like(all_dist, dtype=np.float)
#     for i in range(n):
#         dist[i, all_pred[i, :]] = all_dist[i, :]
#     dist = dist.T
#
#     correct_at_n = np.zeros(len(n_values))
#     #TODO can we do this on the matrix in one go?
#     for qIx, pred in enumerate(predictions):
#         for i,n in enumerate(n_values):
#             # if in top N then also in top NN, where NN > N
#             if np.any(np.in1d(pred[:n], gt[qIx])):
#                 correct_at_n[i:] += 1
#                 break
#     recall_at_n = correct_at_n / qFeat.shape[0]
#
#     recalls = {} #make dict for output
#     for i,n in enumerate(n_values):
#         recalls[n] = recall_at_n[i]
#         print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
#
# #     # db x q
# #     minv = np.min(dist, axis=0)
# #     maxv = np.max(dist, axis=0)
# #     dist = (dist - minv) / (maxv - minv)
#     P, R = drawPR(dist, gtm, True)
#     print("====> AUC : %.4f" % calAvgPred(P, R))
#
#     return dist
#
# def pca_whitening(feat, n_dims):
#     # feat: num x dim
#     # mean of each feature
#     # return: num x n_dims
# #     y = np.zeros((feat.shape[0], n_dims))
#     pca = PCA(n_components=n_dims, whiten=False)
#     y = pca.fit_transform(feat)
#     print("principle component ratio : " + str(np.sum(pca.explained_variance_ratio_)))
#
#     return y
#
#
# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
#
# class L2Norm(nn.Module):
#     def __init__(self, dim=1):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, input):
#         return F.normalize(input, p=2, dim=self.dim)
#
#
# class MyNetVLAD:
#
#     def __init__(self, ckpt, nGPU=1):
#
#         self.encoder_dim = 512
#         self.num_clusters = 64
#         self.vladv2 = True
#         self.nGPU = nGPU
#         self.model = None
#         self.ckpt = ckpt
#
#     def __call__(self):
#
#         if self.model == None:
#             encoder = models.vgg16(pretrained=False)
#             # capture only feature part and remove last relu and maxpool
#             layers = list(encoder.features.children())[:-2]
#             encoder = nn.Sequential(*layers)
#             model = nn.Module()
#             model.add_module('encoder', encoder)
#             net_vlad = netvlad.NetVLAD(num_clusters=self.num_clusters, dim=self.encoder_dim)
#             model.add_module('pool', net_vlad)
#
#             if self.nGPU > 1 and torch.cuda.device_count() > 1:
#                 model.encoder = nn.DataParallel(model.encoder)
#                 model.pool = nn.DataParallel(model.pool)
#
#             # load model
#             checkpoint = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
#             model.load_state_dict(checkpoint['state_dict'])
#             model = model.to(device)
#
#             self.model = model
#
#
#     def genFeatures(self, loader, n):
#         self.model.eval()
#         with torch.no_grad():
#             pool_size = self.encoder_dim * self.num_clusters
#             dbFeat = np.empty((n, pool_size))
#
#             for iteration, (inp, indices) in enumerate(loader, 1):
#                 inp = inp.to(device)
#                 image_encoding = self.model.encoder(inp)
#                 vlad_encoding = self.model.pool(image_encoding)
#
#                 dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
#                 if iteration % 50 == 0 or len(loader) <= 10:
#                     print("==> Batch ({}/{})".format(iteration,
#                         len(loader)), flush=True)
#
#                 del inp, image_encoding, vlad_encoding
#         del loader
#
#         return dbFeat.astype('float32')
#
#
# class MyAlexNet:
#     # output conv5 desc
#     def __init__(self, ckpt, num_classes=1000):
#
#         self.ckpt = ckpt
#         self.num_classes = num_classes
#         self.layer = 8+1
#
#     def __call__(self):
#         features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, self.num_classes),
#         )
#
#         model = nn.Module()
#         model.add_module('features', features)
#         model.add_module('avgpool', avgpool)
#         model.add_module('classifier', classifier)
#
#         checkpoint = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
#         model.load_state_dict(checkpoint)
#         model = model.to(device)
#         self.model = model
#
#
#     def genFeatures(self, loader, n):
#         self.model.eval()
#         with torch.no_grad():
#             for iteration, (inp, indices) in enumerate(loader, 1):
# #                 print(iteration)
#                 inp = inp.to(device)
#                 image_encoding = self.model.features[:self.layer](inp)
#                 image_encoding = image_encoding.view(image_encoding.size(0), -1)
#                 if iteration == 1:
#                     dbFeat = image_encoding.detach().cpu().numpy()
#                 else:
#                     dbFeat = np.concatenate((dbFeat, image_encoding), axis=0)
#
#                 if iteration % 50 == 0 or len(loader) <= 10:
#                     print("==> Batch ({}/{})".format(iteration,
#                         len(loader)), flush=True)
#
#                 del inp, image_encoding
#         del loader
#
#         return dbFeat.astype('float32')

