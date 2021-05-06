import os
import numpy as np
import pickle
from collections import namedtuple
import csv

root = r'E:/project/scut/graduation/datasets/oxford_robotcar/'
# day
data_dir = '2015-05-22-11-14-30/stereo/'
# snow
# data_dir = '2015-02-03-08-45-10/stereo/'
# # night
# data_dir = '2014-12-16-18-44-24/stereo/'

gps_file = root + data_dir + 'gps/gps.csv'
pic_file = root + data_dir + 'centre/'
# save_file = r'night/'
save_file = r'day/'
# save_file = r'snow/'
img_suffix = '.png'
dump_bits = 6

# with open('gps.pkl', 'rb') as f:
#     data = pickle.load(f)

data = {'timestamp' : [], 'latitude' : [], 'longitude' : []}
with open(gps_file) as f:
    f_csv = csv.reader(f)
    Row = namedtuple('Row', next(f_csv))

    for each_row in f_csv:
        row_info = Row(*each_row)
        # 140535xxx the xxx part is compared
        data['timestamp'].append(int(row_info.timestamp[dump_bits:]))
        data['latitude'].append(float(row_info.latitude))
        data['longitude'].append(float(row_info.longitude))


data['timestamp'] = np.array(data['timestamp'], dtype=np.longlong)
data['latitude'] = np.array(data['latitude'], dtype=np.float)
data['longitude'] = np.array(data['longitude'], dtype=np.float)

# with open('gps.pkl', 'wb') as f:
#     pickle.dump(data, f)
#

img_name = os.listdir(pic_file)
img_timeStamp = [int(x[dump_bits:].strip(img_suffix)) for x in img_name]
img_timeStamp = sorted(img_timeStamp)

# img_timeStamp = np.array(img_timeStamp)
# inter = img_timeStamp[1:] - img_timeStamp[:-1]


imgLoc = np.zeros((2, len(img_timeStamp)))  # dim x num

cnt = 0
for k, each in enumerate(img_timeStamp):
    diff = abs(each - data['timestamp'][cnt])
    while cnt < len(data['timestamp']):
        diff1 = abs(each - data['timestamp'][cnt])
        if diff1 > diff:
            cnt -= 1
            imgLoc[:, k] = np.array([data['latitude'][cnt], data['longitude'][cnt]])
            break
        diff = diff1
        cnt += 1

np.savez(save_file + 'gps.npz', gps=imgLoc)
