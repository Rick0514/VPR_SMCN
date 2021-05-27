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
# only show result of MCN, SMCN and SMCNTF
# commented (*) means you need to specify your customized param
# -------------------------------------------------------------------

img_size = 150
img_gap = 10
frame_width = 6

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


def get_a_row(imglist, gtlist):
    """
    imglist: [Reference PW MCN SMCN SMCNTF] the image will be resize afterwards
    gtlist: bool array
    return: a numpy matrix with five images and more anotations.
    """
    num = len(imglist)
    # make a white blank
    row = 255 * np.ones((img_size, num*img_size + (num-1)*img_gap, 3), dtype=np.uint8)

    st = 0
    for k, each in enumerate(imglist):
        each = cv2.resize(each, (img_size, img_size))
        if k > 0:
            each = add_frame(each, gtlist[k-1])
        row[:, st:st+img_size, :] = each
        st += (img_size + img_gap)

    return row

def get_list(S_pw, S_mcn, S_smcn, S_smcntf, picid, gtg, gtb, flag):
    """
    gtg: groundtruth for totally correct( which is green)
    gtb: groundtruth for nearly correct( which is blue)
    """
    id_pw = np.argmax(S_pw[:, picid], axis=0)
    id_mcn = np.argmax(S_mcn[:, picid], axis=0)
    id_smcn = np.argmax(S_smcn[:, picid], axis=0)
    id_smcntf = S_smcntf[flag][picid]

    list1 = [picid, id_pw, id_mcn, id_smcn, id_smcntf]
    gtgl = np.where(gtg[:, picid])[0]
    gtbl = np.where(gtb[:, picid])[0]

    list2 = []
    for each in list1[1:]:
        if each in gtgl:
            list2.append(0)
        elif each in gtbl:
            list2.append(1)
        else:
            list2.append(2)

    return list1, list2


def add_anotation(img):

    ex_r = 50
    ex_c = 100
    # make a white blank
    r, c, _ = img.shape
    row = 255 * np.ones((r + ex_r, c + ex_c, 3), dtype=np.uint8)
    row[ex_r:, ex_c:, :] = img

    r_text = ['NL', 'GP', 'OX', 'Scut']
    c_text = ['Ref', 'PW', 'MCN', 'SMCN', 'SMCNTF']

    for k, each in enumerate(r_text):
        x_st = 20
        y_st = ex_r + k*img_size + img_size // 2 + 10
        cv2.putText(row, each, (x_st, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)

    for k, each in enumerate(c_text):
        x_st = ex_c + k * img_size + img_size // 2 - 25
        y_st = 40
        cv2.putText(row, each, (x_st, y_st), cv2.FONT_HERSHEY_SIMPLEX, 1, vis.black, 2, cv2.LINE_8)

    return row


show_num = 100   # coz the minimum dataset scut only contains 24 images

# nordland row
nd_root = 'E:/project/scut/graduation/datasets/nordland/'   #(*)
# load xxx.npz
S_file = './vis3/nd_sumwin.npz'     #(*)
S = np.load(S_file)
nd_S_pw = S['S_pw']
nd_S_mcn = S['S_mcn']
nd_S_smcn = S['S_smcn']
nd_S_smcntf = S['S_smcntf']
del S_file, S
nd_img_f = '%d.png'
nd_dbfile = nd_root + 'summer/'
nd_qfile = nd_root + 'winter/'
nd_num = nd_S_pw.shape[1]

# gardens point row
gp_root = 'E:/project/scut/graduation/datasets/gardens_point/'   #(*)
S_file = './vis3/gp_dlnr.npz'     #(*)
S = np.load(S_file)
gp_S_pw = S['S_pw']
gp_S_mcn = S['S_mcn']
gp_S_smcn = S['S_smcn']
gp_S_smcntf = S['S_smcntf']
del S_file, S
gp_img_f = 'Image%03d.jpg'
gp_dbfile = gp_root + 'day_left/'
gp_qfile = gp_root + 'night_right/'
gp_num = gp_S_pw.shape[1]

# oxford robotcar row
ox_root = 'E:/project/scut/graduation/datasets/oxford_robotcar/'   #(*)
S_file = './vis3/ox_sn.npz'     #(*)
S = np.load(S_file)
ox_S_pw = S['S_pw']
ox_S_mcn = S['S_mcn']
ox_S_smcn = S['S_smcn']
ox_S_smcntf = S['S_smcntf']
del S_file, S
ox_img_f = '%d.png'
ox_dbfile = ox_root + 'snow/'
ox_qfile = ox_root + 'night/'
ox_num = ox_S_pw.shape[1]

# scut row
sc_root = 'E:/project/scut/graduation/datasets/scut/'   #(*)
S_file = './vis3/scut.npz'     #(*)
S = np.load(S_file)
sc_S_pw = S['S_pw']
sc_S_mcn = S['S_mcn']
sc_S_smcn = S['S_smcn']
sc_S_smcntf = S['S_smcntf']
del S_file, S
sc_img_f = '%d.jpg'
sc_dbfile = sc_root + 'day/'
sc_qfile = sc_root + 'dawn/'
sc_num = sc_S_pw.shape[1]

# error tolerance
nd_err = 9
gp_err = 3
ox_err = 5
sc_err = 1

# make groudtruth
nd_gtg = utils.getGroundTruthMatrix(nd_num, nd_err)
nd_gtb = utils.getGroundTruthMatrix(nd_num, 2*nd_err)
gp_gtg = utils.getGroundTruthMatrix(gp_num, gp_err)
gp_gtb = utils.getGroundTruthMatrix(gp_num, 2*gp_err)
sc_gtg = utils.getGroundTruthMatrix(sc_num, sc_err)
sc_gtb = utils.getGroundTruthMatrix(sc_num, 2*sc_err)
db_gps = np.load(ox_root + 'gps_snow.npy')
q_gps = np.load(ox_root + 'gps_night.npy')
_, ox_gtg = utils.getGpsGT(db_gps, q_gps, ox_err)
_, ox_gtb = utils.getGpsGT(db_gps, q_gps, 2*ox_err)

# start id
nd_start_id = 0
gp_start_id = 0
ox_start_id = 0
sc_start_id = 0

# filter ox gt
tmp = np.sum(ox_gtg, axis=0)
ox_filtered_id = np.where(tmp > 0)[0]

with open('./vis3/pathid.pkl', 'rb') as f:  # (*)
    S_smcntf = pickle.load(f)

# save video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 2.0
video = cv2.VideoWriter('./vis4/backend.mp4', fourcc, fps, (890, 650))

for i in trange(show_num):
    # nd
    imlist, gtlist = get_list(nd_S_pw, nd_S_mcn, nd_S_smcn, S_smcntf, nd_start_id+i, nd_gtg, nd_gtb, 'nd')
    imglist = [cv2.imread(nd_qfile + nd_img_f % i)]
    for each in imlist[1:]:
        imglist.append(cv2.imread(nd_dbfile + nd_img_f % each))
    nd_row = get_a_row(imglist, gtlist)

    # gp
    imlist, gtlist = get_list(gp_S_pw, gp_S_mcn, gp_S_smcn, S_smcntf, gp_start_id+i, gp_gtg, gp_gtb, 'gp')
    imglist = [cv2.imread(gp_qfile + gp_img_f % i)]
    for each in imlist[1:]:
        imglist.append(cv2.imread(gp_dbfile + gp_img_f % each))
    gp_row = get_a_row(imglist, gtlist)

    # ox
    ox_picid = ox_filtered_id[i]
    imlist, gtlist = get_list(ox_S_pw, ox_S_mcn, ox_S_smcn, S_smcntf, ox_picid, ox_gtg, ox_gtb, 'ox')
    imglist = [cv2.imread(ox_qfile + ox_img_f % ox_picid)]
    for each in imlist[1:]:
        imglist.append(cv2.imread(ox_dbfile + ox_img_f % each))
    ox_row = get_a_row(imglist, gtlist)

    # scut
    if i < 24:
        imlist, gtlist = get_list(sc_S_pw, sc_S_mcn, sc_S_smcn, S_smcntf, i, sc_gtg, sc_gtb, 'scut')
        imglist = [cv2.imread(sc_dbfile + sc_img_f % (i * 4))]
        for each in imlist[1:]:
            imglist.append(cv2.imread(sc_qfile + sc_img_f % (each * 4)))
        sc_row = get_a_row(imglist, gtlist)

    row = np.concatenate((nd_row, gp_row, ox_row, sc_row), axis=0)

    if i == 0:
        nrow = add_anotation(row)
    else:
        nrow[50:, 100:, :] = row

    video.write(nrow)

video.release()
    # cv2.imshow('test', nrow)
    # if cv2.waitKey(500) == 27:
    #     cv2.destroyAllWindows()
    #     break

# cv2.waitKey(500)
# cv2.destroyAllWindows()
