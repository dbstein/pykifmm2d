import pyfmmlib2d
import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

"""
Demonstration of the FMM for the Laplace Kernel

If N <= 50000, will do a direct sum and compare to this
Otherwise, will try to call FMMLIB2D through pyfmmlib2d
To compare to
If this fails, no comparison for correctness!

On my macbook pro N=50,000 takes the direct method ~7s, the FMM <1s
(with N_equiv=64, N_cutoff=500)
And gives error <5e-14
"""

random2 = pykifmm2d.utils.random2
Laplace_Kernel_Apply      = pykifmm2d.kernels.laplace.laplace_kernel
Laplace_Kernel_Self_Apply = pykifmm2d.kernels.laplace.laplace_kernel_self
Laplace_Kernel_Form       = pykifmm2d.kernels.laplace.Laplace_Kernel_Form
Prepare_Functions         = pykifmm2d.fmm.prepare_numba_functions

N_total = 100000

# construct some data to run FMM on
N_clusters = 0
N_per_cluster = 1000
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)

# maximum number of points in each leaf of tree for FMM
N_cutoff = 48
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

# get random density
tau = np.random.rand(N_total)/N_total

print('\nLaplace Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get reference solution
reference = True
if N_total <= 50000:
	# by Direct Sum
	st = time.time()
	reference_eval = np.zeros(N_total, dtype=float)
	Laplace_Kernel_Self_Apply(px, py, tau, reference_eval)
	time_direct_eval = (time.time() - st)*1000
	print('\nDirect evaluation took:        {:0.1f}'.format(time_direct_eval))
else:
	# by FMMLIB2D, if available
	try:
		# try:
			# pyfmmlib2d
		# except:
			# import pyfmmlib2d
		source = np.row_stack([px, py])
		st = time.time()
		out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True)
		et = time.time()
		time_fmmlib_eval = (et-st)*1000
		reference_eval = -out['source']['u']/(2*np.pi)
		print('FMMLIB evaluation took:        {:0.1f}'.format(time_fmmlib_eval))
	except:
		print('')
		reference = False

# jit compile internal numba functions
numba_functions = Prepare_Functions(Laplace_Kernel_Apply, Laplace_Kernel_Self_Apply)
# plan fmm
st = time.time()
fmm_plan = pykifmm2d.fmm.fmm_planner(px, py, N_equiv, N_cutoff, Laplace_Kernel_Form, numba_functions, verbose=True)
planning_time = (time.time()-st)*1000
# execute fmm
st = time.time()
fmm_eval = pykifmm2d.fmm.planned_fmm(fmm_plan, tau)
time_fmm_eval = (time.time() - st)*1000
err = np.abs(fmm_eval - reference_eval)

print('FMM planning took:             {:0.1f}'.format(planning_time))
print('FMM evaluation took:           {:0.1f}'.format(time_fmm_eval))
print('\nMaximum difference:            {:0.2e}'.format(err.max()))

"""
import line_profiler
%load_ext line_profiler
%lprun -f pykifmm2d.fmm.planned_fmm pykifmm2d.fmm.planned_fmm(fmm_plan, tau)

import line_profiler
%load_ext line_profiler
%lprun -f pykifmm2d.fmm.fmm_planner fmm_plan = pykifmm2d.fmm.fmm_planner(px, py, N_equiv, N_cutoff, Laplace_Kernel_Form, numba_functions, verbose=True)
"""

# fig, ax = plt.subplots()
# fmm_plan.tree.plot(ax, mpl)

