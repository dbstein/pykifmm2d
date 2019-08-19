import pykifmm2d
import pykifmm2d.class_fmm as fmm
import numpy as np
import numba
import time
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from fast_interp import chebyshev_function_generator
CFG = chebyshev_function_generator.ChebyshevFunctionGenerator

from scipy.special import hankel1
_h0 = CFG(lambda x: hankel1(0, x), 1e-30, 200, tol=1e-14, n=32, verbose=False)
h0 = _h0.get_base_function()

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
Prepare_Functions_OTF  = fmm.prepare_numba_functions_on_the_fly
Prepare_K_Functions    = fmm.Get_Kernel_Functions
helmholtz_k = 1.0

scaleit = 0.25j
# Modified Helmholtz Kernel
@numba.njit(fastmath=True)
def MH_Eval(sx, sy, tx, ty):
    # return _numba_k0(helmholtz_k*np.sqrt((tx-sx)**2 + (ty-sy)**2))*scaleit
    return h0(helmholtz_k*np.sqrt((tx-sx)**2 + (ty-sy)**2))*scaleit

# associated kernel evaluation functions
kernel_functions = Prepare_K_Functions(MH_Eval)
(KF, KA, KAS) = kernel_functions
# jit compile internal numba functions
numba_functions_otf  = Prepare_Functions_OTF (MH_Eval)
# numba_functions_plan = Prepare_Functions_PLAN(MH_Eval)

N_total = 1000*100
test = 'circle' # clustered or circle or uniform

# construct some data to run FMM on
if test == 'uniform':
    px = np.random.rand(N_total)
    py = np.random.rand(N_total)
    rx = np.random.rand(N_total)
    ry = np.random.rand(N_total)
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
    rx = np.random.rand(N_total)
    ry = np.random.rand(N_total)
elif test == 'circle':
    rand_theta = np.random.rand(N_total)*2*np.pi
    px = np.cos(rand_theta)
    py = np.sin(rand_theta)
    rx = np.random.rand(N_total)*2 - 1
    ry = np.random.rand(N_total)*2 - 1
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 50
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

# get random density
tau = (np.random.rand(N_total) + 1j*np.random.rand(N_total))/N_total
tau = tau.astype(complex)

print('\nModified Helmholtz Kernel Direct vs. FMM demonstration with', N_total, 'points.')

# get reference solution
reference = True
if reference:
    if N_total <= 50000:
        # by Direct Sum
        st = time.time()
        self_reference_eval = np.zeros(N_total, dtype=complex)
        KAS(px, py, tau, out=self_reference_eval)
        time_self_eval = (time.time() - st)*1000
        st = time.time()
        target_reference_eval = np.zeros(N_total, dtype=complex)
        KA(px, py, rx, ry, tau, out=target_reference_eval)
        time_target_eval = (time.time() - st)*1000
        print('\nDirect self evaluation took:        {:0.1f}'.format(time_self_eval))
        print('Direct target evaluation took:      {:0.1f}'.format(time_target_eval))
    else:
        # by FMMLIB2D, if available
        try:
            import pyfmmlib2d
            source = np.row_stack([px, py])
            target = np.row_stack([rx, ry])
            st = time.time()
            out = pyfmmlib2d.HFMM(source, target, charge=tau, compute_source_potential=True, compute_target_potential=True, helmholtz_parameter=helmholtz_k)
            self_reference_eval = out['source']['u']
            target_reference_eval = out['target']['u']
            print('FMMLIB self+target evaluation took:   {:0.1f}'.format((time.time()-st)*1000))
            st = time.time()
            out = pyfmmlib2d.HFMM(source, charge=tau, compute_source_potential=True, helmholtz_parameter=helmholtz_k)
            print('FMMLIB self only evaluation took:     {:0.1f}'.format((time.time()-st)*1000))
            st = time.time()
            out = pyfmmlib2d.HFMM(source, target, charge=tau, compute_target_potential=True, helmholtz_parameter=helmholtz_k)
            print('FMMLIB target only evaluation took:   {:0.1f}'.format((time.time()-st)*1000))
        except:
            print('')
            reference = False

# do my FMM (once first, to compile functions...)
FMM = fmm.FMM(px, py, kernel_functions, numba_functions_otf, N_equiv, N_cutoff, True)
FMM.precompute()
FMM.build_expansions(tau)
fmm_eval = FMM.evaluate_to_sources()

st = time.time()
print('')
FMM = fmm.FMM(px, py, kernel_functions, numba_functions_otf, N_equiv, N_cutoff, True)
FMM.precompute()
FMM.build_expansions(tau)
print('pyfmmlib2d generation took:     {:0.1f}'.format((time.time()-st)*1000))
st = time.time()
self_fmm_eval = FMM.evaluate_to_sources()
print('pyfmmlib2d source eval took:    {:0.1f}'.format((time.time()-st)*1000))

target_fmm_eval = FMM.evaluate_to_points(rx, ry)
st = time.time()
target_fmm_eval = FMM.evaluate_to_points(rx, ry)
print('pyfmmlib2d target eval took:    {:0.1f}'.format((time.time()-st)*1000))

if reference:
    self_err = np.abs(self_fmm_eval - self_reference_eval)
    target_err = np.abs(target_fmm_eval - target_reference_eval)
    print('\nMaximum difference, self:       {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:     {:0.2e}'.format(target_err.max()))

if False:
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


