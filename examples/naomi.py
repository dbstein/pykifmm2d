import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

"""
Demonstration of the FMM for the Naomi Kernel

If N <= 50000, will do a direct sum and compare to this
"""

random2 = pykifmm2d.utils.random2
Naomi_Kernel_Apply      = pykifmm2d.kernels.naomi.naomi_kernel_serial
Naomi_Kernel_Self_Apply = pykifmm2d.kernels.naomi.naomi_kernel_self_serial
Naomi_Kernel_Form       = pykifmm2d.kernels.naomi.naomi_kernel_form
Naomi_Kernel_Eval       = pykifmm2d.kernels.naomi.naomi_kernel_eval
Naomi_Kernel_Eval_Many  = pykifmm2d.kernels.naomi.naomi_multi_eval1
Naomi_Test_GF           = pykifmm2d.kernels.naomi.base_gf
Prepare_Functions         = pykifmm2d.fmm.prepare_numba_functions

# jit compile internal numba functions
numba_functions = Prepare_Functions(Naomi_Kernel_Apply, Naomi_Kernel_Self_Apply, Naomi_Kernel_Eval)

N_total = 5000

# construct some data to run FMM on
N_clusters = 0
N_per_cluster = 0
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
px /= 20000
py /= 20000

# first test the gf
xtest = np.concatenate([ (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4), np.linspace(0, 199, 10000)[1:] ])
gfa = Naomi_Test_GF(xtest)
gfb = np.empty_like(gfa)
Naomi_Kernel_Eval_Many(xtest, gfb)
regularizer = np.abs(gfa)
regularizer[regularizer < 1] = 1
err = np.abs(gfa-gfb)/regularizer

print('Error in estimation of GF is: {:0.2e}'.format(err.max()))

# maximum number of points in each leaf of tree for FMM
N_cutoff = 20
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

# get random density
tau = np.random.rand(N_total)/N_total

print('\nNaomi Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get reference solution
if N_total <= 50000:
	# by Direct Sum
	st = time.time()
	reference_eval = np.zeros(N_total, dtype=float)
	Naomi_Kernel_Self_Apply(px, py, tau, reference_eval)
	time_direct_eval = (time.time() - st)*1000
	print('\nDirect evaluation took:        {:0.1f}'.format(time_direct_eval))

# do my FMM
st = time.time()
fmm_eval, tree = pykifmm2d.on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, \
                    Naomi_Kernel_Form, numba_functions, verbose=True)
time_fmm_eval = (time.time() - st)*1000
regularizer = reference_eval.copy()
err = np.abs(fmm_eval - reference_eval)

print('\nMaximum difference:            {:0.2e}'.format(err.max()))




