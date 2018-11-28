import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

"""
Demonstration of the FMM for the Laplace Kernel

Do not chose N_total too large; this code executes a direct sum to get
the true value to compare against!

On my macbook pro N=50,000 takes the direct method ~7s, the FMM ~1s
"""

random2 = pykifmm2d.utils.random2
Laplace_Kernel_Apply = pykifmm2d.kernels.laplace.Laplace_Kernel_Apply
Laplace_Kernel_Form  = pykifmm2d.kernels.laplace.Laplace_Kernel_Form
on_the_fly_fmm = pykifmm2d.on_the_fly_fmm4

N_total = 100000
profile = False

# construct some data to run FMM on
N_clusters = 5
N_per_cluster = 1000
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)

# maximum number of points in each leaf of tree for FMM
N_cutoff = 200
# number of points used in Check/Equivalent Surfaces
N_equiv = 64

# get random density
tau = np.random.rand(N_total)/N_total

print('\nLaplace Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# do direct evaluation
if N_total <= 50000:
	st = time.time()
	direct_eval = Laplace_Kernel_Apply(px, py, charge=tau)
	time_direct_eval = (time.time() - st)*1000
	print('\nDirect evaluation took:        {:0.1f}'.format(time_direct_eval))

x = px
y = py
Kernel_Apply = Laplace_Kernel_Apply
Kernel_Form = Laplace_Kernel_Form
verbose = True
Ncutoff = N_cutoff
Nequiv = N_equiv
import numpy as np
import scipy as sp
import scipy.linalg
import time
from pykifmm2d.tree4 import Tree

def get_level_information(node_width, theta):
    # get information for this level
    dd = 0.1
    r1 = 0.5*node_width*(np.sqrt(2)+dd)
    r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
    small_surface_x_base = r1*np.cos(theta)
    small_surface_y_base = r1*np.sin(theta)
    large_surface_x_base = r2*np.cos(theta)
    large_surface_y_base = r2*np.sin(theta)
    return small_surface_x_base, small_surface_y_base, large_surface_x_base, \
                large_surface_y_base, r1, r2

def classify(node1, node2):
    # for two nodes at the same depth, determine relative position to
    # figure out which of the M2Ls to use
    xdist = int(round((node2.xlow - node1.xlow)/node1.xran))
    ydist = int(round((node2.ylow - node1.ylow)/node1.yran))
    closex = xdist in [-1,0,1]
    closey = ydist in [-1,0,1]
    ilist = not (closex and closey)
    return ilist, xdist, ydist

def generate_kernel_apply(kernel_form):
    def kernel_apply(sx, sy, tau, tx=None, ty=None):
        G = Kernel_Form(sx, sy, tx, ty)
        return G.dot(tau)
    return kernel_apply

def fake_print(*args, **kwargs):
    pass
def get_print_function(verbose):
    return print if verbose else fake_print

st = time.time()
fmm_eval, tree = on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, Laplace_Kernel_Form, Laplace_Kernel_Apply, True)
time_fmm_eval = (time.time() - st)*1000
print('\nFMM evaluation took:           {:0.1f}'.format(time_fmm_eval))

if N_total <= 50000:
	print('Difference: {:0.2e}'.format(np.abs(fmm_eval-direct_eval).max()))

if profile:
	import line_profiler
	%load_ext line_profiler
	%lprun -f pykifmm2d.fmm4._on_the_fly_fmm on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, Laplace_Kernel_Form, Laplace_Kernel_Apply, True)

import pyfmmlib2d
RFMM = pyfmmlib2d.RFMM
source = np.row_stack([px, py])
st = time.time()
out = RFMM(source, charge=tau, compute_source_potential=True)
et = time.time()
time_fmmlib_eval = (et-st)*1000
fmmlib_eval = -out['source']['u']/(2*np.pi)
print('FMMLIB evaluation took:        {:0.1f}'.format(time_fmmlib_eval))
print('Difference: {:0.2e}'.format(np.abs(fmm_eval-fmmlib_eval).max()))



