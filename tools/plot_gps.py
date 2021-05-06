import gmplot
import numpy as np
import matplotlib.pyplot as plt
# Create the map plotter:
apikey = 'xx'   # your API Key

save_path = 'day/'
# save_path = 'snow/'
# save_path = 'night/'
gps = np.load(save_path + 'gps.npz')['gps']
num = gps.shape[1]
center_gps = gps[:, num//2]

gmap = gmplot.GoogleMapPlotter(center_gps[0], center_gps[1], 15, apikey=apikey)

# Highlight some attractions:
st = 0
ed = num
attractions_lats, attractions_lngs = list(gps[0, st:ed]), list(gps[1, st:ed])
gmap.plot(attractions_lats, attractions_lngs,
             color='red', edge_width=7)
gmap.scatter(attractions_lats, attractions_lngs, color='red',
             marker=False, s=5)
gmap.scatter([attractions_lats[0]], [attractions_lngs[0]], color='red',
             label=['S'])

# Draw the map to an HTML file:
gmap.draw(save_path + 'map.html')

# plt.figure()
# plt.scatter(attractions_lats[0], attractions_lngs[0], s=100, marker='*', c='red')
# plt.scatter(attractions_lats, attractions_lngs, s=5, c='blue')
