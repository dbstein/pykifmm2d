try:
	import pyfmmlib2d
except:
	pass
import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

"""
Demonstration of the FMM for the Modified Helmholtz Kernel

If N <= 50000, will do a direct sum and compare to this
Otherwise, will try to call FMMLIB2D through pyfmmlib2d
To compare to
If this fails, no comparison for correctness!

On my macbook pro N=50,000 takes the direct method ~7s, the FMM <1s
(with N_equiv=64, N_cutoff=500)
And gives error <5e-14
"""

random2 = pykifmm2d.utils.random2
MH_get = pykifmm2d.kernels.modified_helmholtz.generate_modified_helmholtz_functions
Prepare_Functions = pykifmm2d.fmm.prepare_numba_functions

N_total = 1000000
helmholtz_k = 0.1

# construct some data to run FMM on
N_clusters = 50
N_per_cluster = 1000
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)

# maximum number of points in each leaf of tree for FMM
N_cutoff = 50
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

Tree = pykifmm2d.Tree
st = time.time()
tree = Tree(px, py, 50)
print('Took: {:0.3f}'.format((time.time()-st)*1000))

from pykifmm2d.tree import split_bad_leaves
import line_profiler
%load_ext line_profiler
%lprun -f Tree.__init__ Tree(px, py, 50)
%lprun -f split_bad_leaves Tree(px, py, 50)








