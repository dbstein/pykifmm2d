import pykifmm2d
import pykifmm2d.svd_fmm
import numpy as np
import numba
import time
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from pykifmm2d.misc.numba_special_functions import _numba_k0

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
Prepare_Functions_OTF  = pykifmm2d.fmm.prepare_numba_functions_on_the_fly
Prepare_Functions_PLAN = pykifmm2d.fmm.prepare_numba_functions_planned
Prepare_K_Functions    = pykifmm2d.fmm.Get_Kernel_Functions
helmholtz_k = 1.0

scaleit = 1.0 / (2*np.pi)
# Modified Helmholtz Kernel
@numba.njit("f8(f8,f8,f8,f8)", fastmath=True)
def MH_Eval(sx, sy, tx, ty):
    return _numba_k0(helmholtz_k*np.sqrt((tx-sx)**2 + (ty-sy)**2))*scaleit

# associated kernel evaluation functions
kernel_functions = Prepare_K_Functions(MH_Eval)
(KF, KA, KAS) = kernel_functions
# jit compile internal numba functions
numba_functions_otf  = Prepare_Functions_OTF (MH_Eval)
numba_functions_plan = Prepare_Functions_PLAN(MH_Eval)

N_total = 1000*100
test = 'circle' # clustered or circle or uniform

# construct some data to run FMM on
if test == 'uniform':
    px = np.random.rand(N_total)
    py = np.random.rand(N_total)
elif test == 'clustered':
    N_clusters = 10
    N_per_cluster = int((N_total / N_clusters))
    N_random = N_total - N_clusters*N_per_cluster
    center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
    px, py = random2(N_total, -1, 1)
    px[:N_random] *= 100
    py[:N_random] *= 100
    px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
    py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
    px /= 100
    py /= 100
elif test == 'circle':
    rand_theta = np.random.rand(N_total)*2*np.pi
    px = np.cos(rand_theta)
    py = np.sin(rand_theta)
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 200
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

# get random density
tau = np.random.rand(N_total)/N_total

print('\nModified Helmholtz Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get reference solution
reference = True
if reference:
    # if N_total <= 50000:
    if False:
        # by Direct Sum
        st = time.time()
        reference_eval = np.zeros(N_total, dtype=float)
        KAS(px, py, tau, out=reference_eval)
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
            reference_eval = out['source']['u'].real
            print('FMMLIB evaluation took:        {:0.1f}'.format(time_fmmlib_eval))
        except:
            print('')
            reference = False

# do my FMM
st = time.time()
fmm_eval, tree = pykifmm2d.svd_fmm.on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, \
                    kernel_functions, numba_functions_otf, verbose=True)
time_fmm_eval = (time.time() - st)*1000
if reference:
    err = np.abs(fmm_eval - reference_eval)
    print('\nMaximum difference:            {:0.2e}'.format(err.max()))





if True:
    # plan fmm
    st = time.time()
    fmm_plan = pykifmm2d.fmm.fmm_planner(px, py, N_equiv, N_cutoff, kernel_functions, numba_functions_plan, verbose=True)
    planning_time = (time.time()-st)*1000
    # execute fmm
    st = time.time()
    fmm_eval = pykifmm2d.fmm.planned_fmm(fmm_plan, tau)
    time_fmm_eval = (time.time() - st)*1000
    err = np.abs(fmm_eval - reference_eval)

    print('\nFMM planning took:               {:0.1f}'.format(planning_time))
    print('FMM evaluation took:             {:0.1f}'.format(time_fmm_eval))
    print('Maximum difference:              {:0.2e}'.format(err.max()))
