import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
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
N_clusters = 20
N_per_cluster = 2000
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

print('\nModified Helmholtz Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get helmholtz functions
MH_Kernel_Form, MH_Kernel_Apply, MH_Kernel_Self_Apply = MH_get(helmholtz_k)

# get reference solution
reference = True
if N_total <= 50000:
	# by Direct Sum
	st = time.time()
	reference_eval = np.zeros(N_total, dtype=float)
	MH_Kernel_Self_Apply(px, py, tau, reference_eval)
	time_direct_eval = (time.time() - st)*1000
	print('\nDirect evaluation took:        {:0.1f}'.format(time_direct_eval))
else:
	# by FMMLIB2D, if available
	try:
		import pyfmmlib2d
		source = np.row_stack([px, py])
		st = time.time()
		out = pyfmmlib2d.HFMM(source, charge=tau, compute_source_potential=True, helmholtz_parameter=1j*helmholtz_k)
		et = time.time()
		time_fmmlib_eval = (et-st)*1000
		reference_eval = out['source']['u']
		print('FMMLIB evaluation took:        {:0.1f}'.format(time_fmmlib_eval))
	except:
		print('')
		reference = False

# jit compile internal numba functions
numba_functions = Prepare_Functions(MH_Kernel_Apply, MH_Kernel_Self_Apply)
# do my FMM
st = time.time()
fmm_eval, tree = pykifmm2d.on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, \
                    MH_Kernel_Form, numba_functions, verbose=True)
time_fmm_eval = (time.time() - st)*1000
err = np.abs(fmm_eval*0.5/np.pi - reference_eval)

print('\nMaximum difference:            {:0.2e}'.format(err.max()))
