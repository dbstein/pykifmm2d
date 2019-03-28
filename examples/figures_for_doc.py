import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

random2 = pykifmm2d.utils.random2

N_total = 500

# construct some data to run FMM on
N_clusters = 1
N_per_cluster = 480
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = 1.0, 1.0
px, py = random2(N_total, -1, 1)
px[:N_random] *= 10
py[:N_random] *= 10
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)

# maximum number of points in each leaf of tree for FMM
N_cutoff = 100
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

Tree = pykifmm2d.Tree
st = time.time()
tree = Tree(px, py, 50)
print('Took: {:0.3f}'.format((time.time()-st)*1000))

fig, ax = plt.subplots()
tree.plot(ax, mpl, points=True)

for i in range(tree.levels):
	fig1, ax1 = plt.subplots()
	tree.plot_level(ax1, mpl, i+1, color='black', linewidth=10)

