import gmplot
import numpy as np
import matplotlib.pyplot as plt
# Create the map plotter:
apikey = 'AIzaSyAw7RzmFMdMHREo7VZ2e9o-2voreJ9cKdY' # (your API key here)

# path = 'day/'
save_path = ['night/', 'snow/']

gps = np.load(save_path[0] + 'gps.npz')['gps']
num = gps.shape[1]
center_gps = gps[:, num // 2]
del gps

gmap = gmplot.GoogleMapPlotter(center_gps[0], center_gps[1], 15, apikey=apikey)

rcolor = ['red', 'blue']
for k, each_gps_file in enumerate(save_path):
    gps = np.load(each_gps_file + 'gps.npz')['gps']
    attractions_lats, attractions_lngs = list(gps[0, :]), list(gps[1, :])
    gmap.scatter(attractions_lats, attractions_lngs, color=rcolor[k],
                 marker=False, s=5)
    gmap.scatter([attractions_lats[0]], [attractions_lngs[0]], color=rcolor[k],
                 label=['S'])

# Draw the map to an HTML file:
gmap.draw('map.html')

# num = gps.shape[1]
# center_gps = gps[:, num//2]
#
#
# # Highlight some attractions:
# st = 102
# ed = num
#
# plt.figure()
# plt.plot(attractions_lats[0], attractions_lngs[0], s=100, marker='*', c='red')
# plt.plot(attractions_lats, attractions_lngs, s=5, c='blue')
