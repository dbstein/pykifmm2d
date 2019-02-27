try:
	import pyfmmlib2d
except:
	pass
import pykifmm2d
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

random2 = pykifmm2d.utils.random2
MH_get = pykifmm2d.kernels.modified_helmholtz.generate_modified_helmholtz_functions
Prepare_Functions = pykifmm2d.fmm.prepare_numba_functions

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

# which kernel to use ('Laplace', 'Modified Helmholtz', 'Biharmonic', 'Inverse Distance Squared')
problem = 'Laplace'
# for modified helmholtz, set the helmholtz parameter
helmholtz_k = 1
# number of points in test sum
N_total = 100000
# maximum number of points in each leaf of tree for FMM
N_cutoff = 200
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

if problem == 'Laplace':
	Kernel_Apply      = pykifmm2d.kernels.laplace.laplace_kernel
	Kernel_Self_Apply = pykifmm2d.kernels.laplace.laplace_kernel_self
	Kernel_Form       = pykifmm2d.kernels.laplace.Laplace_Kernel_Form
	Kernel_Eval       = pykifmm2d.kernels.laplace.laplace_eval
elif problem == 'Modified Helmholtz':
	MH_get = pykifmm2d.kernels.modified_helmholtz.generate_modified_helmholtz_functions
	Kernel_Form, Kernel_Apply, Kernel_Self_Apply, Kernel_Eval = MH_get(helmholtz_k)
elif problem == 'Biharmonic':
	Kernel_Apply      = pykifmm2d.kernels.biharmonic.biharmonic_kernel
	Kernel_Self_Apply = pykifmm2d.kernels.biharmonic.biharmonic_kernel_self
	Kernel_Form       = pykifmm2d.kernels.biharmonic.Biharmonic_Kernel_Form
	Kernel_Eval       = pykifmm2d.kernels.biharmonic.biharmonic_eval
elif problem == 'Inverse Distance Squared':
	Kernel_Apply      = pykifmm2d.kernels.one_over_d2.ood2_kernel
	Kernel_Self_Apply = pykifmm2d.kernels.one_over_d2.ood2_kernel_self
	Kernel_Form       = pykifmm2d.kernels.one_over_d2.ood2_Kernel_Form
	Kernel_Eval       = pykifmm2d.kernels.one_over_d2.ood2_eval
else:
	raise Exception('Problem '+problem+' not defined.')

# construct some data to run FMM on
N_per_cluster = 1000
N_clusters = int(0.2*N_total / N_per_cluster)
N_random = N_total - N_clusters*N_per_cluster
center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
px, py = random2(N_total, -1, 1)
px[:N_random] *= 100
py[:N_random] *= 100
px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
px /= 100
py /= 100

# get random density
tau = np.random.rand(N_total)/N_total

print('\n', problem, 'Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get reference solution
reference = True
if N_total <= 50000:
	# by Direct Sum
	st = time.time()
	reference_eval = np.zeros(N_total, dtype=float)
	Kernel_Self_Apply(px, py, tau, reference_eval)
	time_direct_eval = (time.time() - st)*1000
	print('\nDirect evaluation took:        {:0.1f}'.format(time_direct_eval))
# by FMMLIB2D, if available
try:
	source = np.row_stack([px, py])
	st = time.time()
	if problem == 'Laplace':
		out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True)
		reference_eval = -out['source']['u']/(2*np.pi)
	elif problem == 'Modified Helmholtz':
		out = pyfmmlib2d.HFMM(source, charge=tau, compute_source_potential=True, helmholtz_parameter=1j*helmholtz_k)
		reference_eval = out['source']['u']*2*np.pi
	elif problem == 'Biharmonic':
		out = pyfmmlib2d.BFMM(source, charge=tau.astype(complex), compute_source_velocity=True)
	else:
		raise Exception('No FMMLIB function provided.')
	et = time.time()
	time_fmmlib_eval = (et-st)*1000
	print('FMMLIB evaluation took:        {:0.1f}'.format(time_fmmlib_eval))
except:
	pass
try:
	reference_eval
	have_reference = True
except:
	print('Selected size too large for direct eval and FMMLIB not available.')
	have_reference = False

# jit compile internal numba functions
numba_functions = Prepare_Functions(Kernel_Apply, Kernel_Self_Apply, Kernel_Eval)
# do on-the-fly FMM
st = time.time()
otf_fmm_eval, tree = pykifmm2d.on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, \
                    Kernel_Form, numba_functions, verbose=True)
time_fmm_eval = (time.time() - st)*1000
if have_reference:
	err = np.abs(otf_fmm_eval - reference_eval)
	print('\nMaximum difference:            {:0.2e}'.format(err.max()))

# plan fmm
st = time.time()
fmm_plan = pykifmm2d.fmm.fmm_planner(px, py, N_equiv, N_cutoff, Kernel_Form, numba_functions, verbose=True)
planning_time = (time.time()-st)*1000
# execute fmm
st = time.time()
plan_fmm_eval = pykifmm2d.fmm.planned_fmm(fmm_plan, tau)
time_fmm_eval = (time.time() - st)*1000

print('\nFMM planning took:               {:0.1f}'.format(planning_time))
print('FMM evaluation took:             {:0.1f}'.format(time_fmm_eval))
if have_reference:
	err = np.abs(plan_fmm_eval - reference_eval)
	print('Maximum difference:              {:0.2e}'.format(err.max()))

"""
import line_profiler
%load_ext line_profiler
%lprun -f pykifmm2d.fmm.planned_fmm pykifmm2d.fmm.planned_fmm(fmm_plan, tau)

import line_profiler
%load_ext line_profiler
%lprun -f pykifmm2d.fmm.fmm_planner fmm_plan = pykifmm2d.fmm.fmm_planner(px, py, N_equiv, N_cutoff, MH_Kernel_Form, numba_functions, verbose=True)
"""

