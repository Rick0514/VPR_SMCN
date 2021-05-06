import numpy as np
import math
import os


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


root = r'E:/project/scut/graduation/datasets/oxford_robotcar/'

dbName = 'day'
qName = 'night'

dbgps = np.load(root + 'gps_' + dbName + '.npy')
qgps = np.load(root + 'gps_' + qName + '.npy')

err = 4
gt, gtm = getGpsGT(dbgps, qgps, err)

