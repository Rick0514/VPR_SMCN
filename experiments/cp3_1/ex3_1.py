import numpy as np
import utils
import MCN, SMCN
import matplotlib.pyplot as plt
import pickle

# experiment 1
# use gardens point datatset to giev the result of MCN's
# unstability when input connections is less
# and the more input connections the more time consumes
# (*) means the params you should specify 

D1 = np.load('./desc/day_right_desc.npy')       #(*)
D2 = np.load('./desc/night_right_desc.npy')     #(*)

D1 = D1 / np.linalg.norm(D1, axis=0)
D2 = D2 / np.linalg.norm(D2, axis=0)

err = 3     #(*)
GT = utils.makeGT(D1.shape[0], err)
nConPerCol = [100, 300, 500, 700, 1000, 1500, 2000]     #(*)

# traverse all nConPerCol and compute the AP and time consumption
# and save dict which has key: ap and time
# if the result is saved, you can comment the following code
# params = MCN.MCNParams(probAddCon=0.05,
#                        nCellPerCol=32,
#                        nConPerCol=200,
#                        minColActivity=0.7,
#                        nColPerPattern=50,
#                        kActiveCol=100)
#
# MCN_time_cost = []
# MCN_ap = []
#
# for each_con in nConPerCol:
#     params.nConPerCol = each_con
#     each_mcn_ap = []
#     each_mcn_time_cost = []
#     for i in range(5):
#         a1, a2 = MCN.runMCN(params, D1, D2, GT)
#         each_mcn_ap.append(a1)
#         each_mcn_time_cost.append(a2)
#         print('con %d, %d / 5 finished' % (each_con, i+1))
#
#     MCN_ap.append(each_mcn_ap)
#     MCN_time_cost.append(each_mcn_time_cost)
# data = dict()
# data['ap'] = MCN_ap
# data['time'] = MCN_time_cost
# with open('./experiments/cp3_1/mcn.pkl', 'wb') as f:  #(*)
#     pickle.dump(data, f)

# if above result is saved, you can uncomment the following code
with open('./experiments/cp3_1/mcn.pkl', 'rb') as f:    #(*)
    data = pickle.load(f)

MCN_ap = data['ap']
MCN_time_cost = data['time']

SMCNTF_ap, SMCNTF_time_cost = SMCN.runSMCN(D1, D2, GT, cannum=5, err=err)   #(*) specify the cannum

x1 = nConPerCol
x2 = [[xx] * 5 for xx in x1]
y1 = np.mean(np.array(MCN_time_cost), axis=1)
y2 = MCN_ap
yy2 = np.mean(np.array(MCN_ap), axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
le1 = ax1.scatter(x2, y2, color=utils.color[0])
le2, = ax1.plot(x1, yy2, color=utils.color[1], linewidth=2)
le3, = ax1.plot([0, 2000], [SMCNTF_ap]*2, color=utils.color[1], linestyle='-.', linewidth=2)
ax1.set_ylabel('AP')
ax1.set_ylim([0.4, 0.8])
# ax1.set_title("Double Y axis")
ax2 = ax1.twinx()  # this is the important function
le4, = ax2.plot(x1, y1, color=utils.color[2], linewidth=2)
le5, = ax2.plot([0, 2000], [SMCNTF_time_cost]*2, color=utils.color[2], linestyle='-.', linewidth=2)
ax2.set_xlim([0, 2000])
ax2.set_ylabel('time/s')
ax1.set_xlabel('pattern dimensions')
ax1.grid()
ax2.text(x=1000,#文本x轴坐标
         y=10, #文本y轴坐标
         s='only cost 0.235s')
ax1.legend([le1, le2, le3, le4, le5], ['MCN test 5 times', 'MCN AP', 'SMCN+TF AP', 'MCN time cost', 'SMCN+TF time cost'], loc=2)
plt.show()
