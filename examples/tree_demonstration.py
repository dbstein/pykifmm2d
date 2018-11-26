import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

random2 = pykifmm2d.utils.random2

# construct some data to tree
N_total = 1000
N_random = int(0.1*N_total)
N_per_cluster = int(round((N_total - N_random)/20))
N_clusters = 20
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)

# maximum number of points in each leaf of tree
N_cutoff = 100

class empty(object):
	def __init__(self):
		pass

# generate tree
st = time.time()
tree = pykifmm2d.Tree(px, py, N_cutoff)
time_tree = (time.time() - st)*1000

print('\nTree for', N_total, 'points formed in {:0.1f}'.format(time_tree), 'ms.')
print('Tree structure:')
tree.print_structure()
if N_total <= 500000:
	fig, ax = plt.subplots()
	tree.plot(ax, mpl, points=True, s=0.005)

# import line_profiler
# %load_ext line_profiler
# %lprun -f tree.__init__ pykifmm2d.Tree(px, py, N_cutoff)

