import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from matplotlib.pyplot import MultipleLocator
import main.utils as utils
import tools.visualize as vis

# ---------------------------- info ----------------------------------
# visualize hyper parameters
# you will need the result file .npy generated by ex3_3.py
# code end with comment (*) means you should specify
# ---------------------------- info ----------------------------------

dataset_name = ['nordland', 'gardens_point', 'ox_robotcar']
net = ['alexnet', 'netvlad']

k1 = [2.0, 3.0, 4.0, 5.0]
k2 = [0.03, 0.05, 0.07, 0.09]

x, y = np.meshgrid(k1, k2)
x = x.ravel()
y = y.ravel()

res = np.load('./' + net[1] + '_' + dataset_name[2] + '.npy')   #(*)
res = res.ravel()

top3id = np.argsort(-res)[:3]

x3 = x[top3id]
y3 = y[top3id]
res3 = res[top3id]

params = list(zip(x3, y3))
for i in range(3):
    print(str(params[i]) + ' : ' + str(res3[i]))


bot = np.zeros((4, 4)).ravel()
w = 0.15
d = 0.004

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

color = ['#ff0000', '#7fff00']
alpha = [0.3, 0.4]
for i in range(2):
    if i == 0:
        xx = x - w / 2
        yy = y - d / 2
    else:
        xx = x + w / 2
        yy = y + d / 2

    res1 = np.load('./' + net[i] + '_' + dataset_name[0] + '.npy')
    res1 = res1.ravel()
    ax1.bar3d(xx, yy, bot, w, d, res1, color=color[i], alpha=alpha[i])

    res2 = np.load('./' + net[i] + '_' + dataset_name[1] + '.npy')
    res2 = res2.ravel()
    ax2.bar3d(xx, yy, bot, w, d, res2, color=color[i], alpha=alpha[i])

    res3 = np.load('./' + net[i] + '_' + dataset_name[2] + '.npy')
    res3 = res3.ravel()
    ax3.bar3d(xx, yy, bot, w, d, res3, color=color[i], alpha=alpha[i])


ax1.set_xlabel('k1', fontsize=vis.font_text)
ax1.set_ylabel('k2', fontsize=vis.font_text)
ax1.set_zlabel('AP', fontsize=vis.font_text)
ax1.set_title('Nordland', fontsize=vis.font_text+5)
ax1.set_xticks(k1)
ax1.set_yticks(k2)
ax1.view_init(20, 130)
ax1.tick_params(labelsize=vis.font_text)


ax2.set_xlabel('k1', fontsize=vis.font_text)
ax2.set_ylabel('k2', fontsize=vis.font_text)
ax2.set_zlabel('AP', fontsize=vis.font_text)
ax2.set_title('Gardens Point', fontsize=vis.font_text+5)
ax2.set_xticks(k1)
ax2.set_yticks(k2)
ax2.view_init(20, 130)
ax2.tick_params(labelsize=vis.font_text)


ax3.set_xlabel('k1', fontsize=vis.font_text)
ax3.set_ylabel('k2', fontsize=vis.font_text)
ax3.set_zlabel('AP', fontsize=vis.font_text)
ax3.set_title('Oxford Robotcar', fontsize=vis.font_text+5)
ax3.set_xticks(k1)
ax3.set_yticks(k2)
ax3.view_init(20, 130)
ax3.tick_params(labelsize=vis.font_text)
gap = MultipleLocator(0.04)
ax3.zaxis.set_major_locator(gap)

