from os.path import join, exists
from os import listdir
import cv2
import numpy as np
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic


data_name = 'day'
file = data_name
# night_file = r'./night'
data = np.load(join(file, 'cgps.npy'))
gps = data[2:, :]
idx = data[0, :].astype(np.int)

# night_data = np.load(join(night_file, 'cgps.npy'))
# night_gps = night_data[2:, :]
# night_id = night_data[0, :].astype(np.int)
# night_dist = night_data[1, :]

root = r'E:/project/scut/graduation/datasets/oxford_robotcar/'
# day
data_dir = '2015-05-22-11-14-30/stereo/'
# snow
# data_dir = '2015-02-03-08-45-10/stereo/'
# night
# data_dir = '2014-12-16-18-44-24/stereo/'
save_file = root + data_name + '/'
gps_file = root + data_dir + 'gps/gps.csv'
pic_file = root + data_dir + 'centre/'
# save_file = r'night/'
# save_file = r'day/'
img_suffix = '.png'
dump_bits = 6


img_name = listdir(pic_file)
img_postfix = img_name[0][:dump_bits]
img_timeStamp = [int(x[dump_bits:].strip(img_suffix)) for x in img_name]
img_timeStamp = sorted(img_timeStamp)

sorted_imgname = [img_postfix + str(x) + img_suffix for x in np.array(img_timeStamp)[idx]]

for k, each in enumerate(sorted_imgname):
    img = join(pic_file, each)
    if exists(img):
        img = Image.open(img)
        img = demosaic(img, 'GBRG')
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(save_file + str(k) + img_suffix, img)

    else:
        print(img + 'is not found')
        break

    if k % 10 == 0:
        print('%d / %d' % (k, len(sorted_imgname)))

# save ground truth
np.save(root + 'gps_' + data_name + '.npy', gps)

