import numpy as np
import main.utils as utils
import cv2
import os
from random import sample
import pickle
import tools.visualize as vis
from tqdm import trange

# ---------------------------- info ----------------------------------
# visualize result in the form of video, which make it more intuitive.
# show result of panoramic case, onmi-SCUT dataset is used.
# compare full desc, 1% desc, 1% + MCN, 1% + SMCN, 1% + SMCNTF
# commented (*) means you need to specify your customized param
# -------------------------------------------------------------------

img_size = 150
frame_width = 6

ex_r = 50
ex_c = 100

# create a canvas first

r_size = img_size*3 + ex_r * 3
c_size = img_size*5 + ex_c + ex_r
canvas = 255 * np.ones((r_size, c_size, 3), dtype=np.uint8)

# puttext
row_text = ['Ref.', 'Full', '1%']
for k, each in enumerate(row_text):
    x_st = 20
    y_st = (1+k)*ex_r + k * img_size + img_size // 2 + 10
    cv2.putText(canvas, each, (x_st, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)

y_st = 40
x_st = ex_c + img_size // 2
cv2.putText(canvas, 'Single', (x_st-50, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'Panoramic', (x_st + ex_r + 2*img_size, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)

y_st = 40 + img_size + ex_r
x_st = ex_c + img_size // 2
cv2.putText(canvas, 'PW', (x_st-25, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'PW', (x_st + ex_r + 2*img_size+50, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)

y_st = 40 + 2*(img_size + ex_r)
x_st = ex_c + img_size // 2
cv2.putText(canvas, 'PW', (x_st-25, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'PW', (x_st + ex_r + img_size-20, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'MCN', (x_st + ex_r + 2*img_size-35, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'SMCN', (x_st + ex_r + 3*img_size-45, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)
cv2.putText(canvas, 'SMCNTF', (x_st + ex_r + 4*img_size-55, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)


def add_frame(img, flag):
    """
    add a red or green frame to img for anotation.
    flag: 0 --> totally correct( within error tolerance )
    1 --> nearly correct ( within 2x error tolerance)
    2 --> wrong
    """
    r, c, _ = img.shape
    vimg = img.copy()
    if flag == 0:
        color = np.array([[[0, 255, 0]]], dtype=np.uint8)
    elif flag == 1:
        color = np.array([[[255, 0, 0]]], dtype=np.uint8)
    else:
        color = np.array([[[0, 0, 255]]], dtype=np.uint8)

    vimg[:frame_width, :, :] = color
    vimg[r-frame_width:, :, ] = color
    vimg[:, :frame_width, ] = color
    vimg[:, c-frame_width:, ] = color

    return vimg

def set_img(img, x, y):
    img = cv2.resize(img, (img_size, img_size))
    canvas[y:y+img_size, x:x+img_size, :] = img

def get_framed_img(img, i, id, gtg, gtb):
    pickid = id[i]
    gtgl = np.where(gtg[:, i])[0]
    gtbl = np.where(gtb[:, i])[0]

    if pickid in gtgl:
        return add_frame(img, 0)
    elif pickid in gtbl:
        return add_frame(img, 1)
    else:
        return add_frame(img, 2)




show_num = 24   # coz the minimum dataset scut only contains 24 images

# scut row
sc_root = 'E:/project/scut/graduation/datasets/scut/'   #(*)
sc_img_f = '%d.jpg'
sc_dbfile = sc_root + 'day/'
sc_qfile = sc_root + 'dawn/'
sc_num = len(os.listdir(sc_dbfile))

# load id file
with open('./id_dict.pkl', 'rb') as f:
    id_dict = pickle.load(f)
sig_f_id = id_dict['sig_f_id']
omni_f_id = id_dict['onmi_f_id']
sig_t_id = id_dict['sig_t_id']
omni_t_id = id_dict['onmi_t_id']
omni_t_mcn_id = id_dict['onmi_t_mcn_id']
omni_t_smcn_id = id_dict['onmi_t_smcn_id']
omni_t_smcntf_id = id_dict['onmi_t_smcntf_id']


# error tolerance
sc_err = 1

# make groudtruth
sc_gtg = utils.getGroundTruthMatrix(sc_num, sc_err)
sc_gtb = utils.getGroundTruthMatrix(sc_num, 2*sc_err)

# save video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 1.0
video = cv2.VideoWriter('./omni.mp4', fourcc, fps, (900, 600))

for i in trange(show_num):

    # first row
    st = 4*i
    img = cv2.imread(sc_qfile + sc_img_f % st)
    set_img(img, ex_c, ex_r)
    set_img(img, ex_c + img_size + ex_r, ex_r)

    for k in range(1,4):
        img = cv2.imread(sc_qfile + sc_img_f % (st + k))
        set_img(img, ex_c + (1+k)*img_size + ex_r, ex_r)

    # second row
    pic = sig_f_id[i] * 4
    img = cv2.imread(sc_dbfile + sc_img_f % pic)
    img = get_framed_img(img, i, sig_f_id, sc_gtg, sc_gtb)
    set_img(img, ex_c, ex_r*2 + img_size)

    pic = omni_f_id[i] * 4
    img = cv2.imread(sc_dbfile + sc_img_f % pic)
    img = get_framed_img(img, i, omni_f_id, sc_gtg, sc_gtb)
    set_img(img, ex_c + img_size*2 + img_size//2 + ex_r, ex_r * 2 + img_size)

    # third row
    pic = sig_t_id[i] * 4
    img = cv2.imread(sc_dbfile + sc_img_f % pic)
    img = get_framed_img(img, i, sig_t_id, sc_gtg, sc_gtb)
    set_img(img, ex_c, ex_r*3 + img_size*2)

    id_list = [omni_t_id, omni_t_mcn_id, omni_t_smcn_id, omni_t_smcntf_id]
    for k in range(4):
        pic = id_list[k][i] * 4
        img = cv2.imread(sc_dbfile + sc_img_f % pic)
        img = get_framed_img(img, i, id_list[k], sc_gtg, sc_gtb)
        set_img(img, ex_c + (1+k)*img_size + ex_r, ex_r*3 + img_size*2)

    video.write(canvas)

video.release()
#     cv2.imshow('test', canvas)
#
#     if cv2.waitKey(500) == 27:
#         cv2.destroyAllWindows()
#         break
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
