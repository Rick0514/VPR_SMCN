import math
import numpy as np

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


save_path = 'day/'
# save_path = 'snow/'
# save_path = 'night/'
gps = np.load(save_path + 'gps.npz')['gps']
num = gps.shape[1]
start_id = 20


dist = []
for i in range(start_id, num-1):
    tmp = haversine(gps[:, i], gps[:, i+1])
    dist.append(tmp)

acu_dist = np.array(dist)
for i in range(1, len(dist)):
    acu_dist[i] = acu_dist[i-1] + dist[i]

# every 2m sample
sample_id = [start_id]
dis = [0]
tmp = 0
tmp1 = 0
for i in range(len(dist)):
    tmp += dist[i]
    tmp1 += dist[i]
    if tmp >= 2:
        sample_id.append(start_id + 1 + i)
        dis.append(tmp1)
        tmp = 0

cgps = np.zeros((4, len(sample_id)))
cgps[0, :] = np.array(sample_id)
cgps[1, :] = np.array(dis)
cgps[2:, :] = gps[:, sample_id]

np.save(save_path + 'cgps.npy', cgps)