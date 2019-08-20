import pykifmm2d
import pykifmm2d.class_fmm as fmm
import numpy as np
import numba
import time
import os

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

cpu_num = int(os.cpu_count()/2)
numba.config.NUMBA_NUM_THREADS = cpu_num
import mkl
mkl.set_num_threads(cpu_num)

random2 = pykifmm2d.utils.random2
Prepare_Functions_OTF  = fmm.prepare_numba_functions_on_the_fly
Prepare_K_Functions    = fmm.Get_Kernel_Functions

# Laplace Kernel
@numba.njit(fastmath=True)
def Laplace_Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx**2 + dy**2
    return -np.log(d2)/(4*np.pi)

# associated kernel evaluation functions
kernel_functions = Prepare_K_Functions(Laplace_Eval)
(KF, KA, KAS) = kernel_functions
# jit compile internal numba functions
numba_functions_otf  = Prepare_Functions_OTF (Laplace_Eval)
# numba_functions_plan = Prepare_Functions_PLAN(Laplace_Eval)

N_source = 1000*100
N_target = 1000*1000*10
test = 'circle' # clustered or circle or uniform

# construct some data to run FMM on
if test == 'uniform':
    px = np.random.rand(N_source)
    py = np.random.rand(N_source)
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
elif test == 'clustered':
    N_clusters = 10
    N_per_cluster = int((N_source / N_clusters))
    N_random = N_source - N_clusters*N_per_cluster
    center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
    px, py = random2(N_source, -1, 1)
    px[:N_random] *= 100
    py[:N_random] *= 100
    px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
    py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
    px /= 100
    py /= 100
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
elif test == 'circle':
    rand_theta = np.random.rand(N_source)*2*np.pi
    px = np.cos(rand_theta)
    py = np.sin(rand_theta)
    rx = np.random.rand(N_target)*2 - 1
    ry = np.random.rand(N_target)*2 - 1
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 50
# number of points used in Check/Equivalent Surfaces
N_equiv = 48

# get random density
tau = (np.random.rand(N_source))

print('\nLaplace FMM with', N_source, 'source pts and', N_target, 'target pts.')

# get reference solution
reference = True
if reference:
    if N_source*N_target <= 10000**2:
        # by Direct Sum
        st = time.time()
        self_reference_eval = np.zeros(N_source, dtype=complex)
        KAS(px, py, tau, out=self_reference_eval)
        time_self_eval = (time.time() - st)*1000
        st = time.time()
        target_reference_eval = np.zeros(N_target, dtype=complex)
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
            dumb_targ = np.row_stack([np.array([0.6, 0.6]), np.array([0.5, 0.5])])
            st = time.time()
            out = pyfmmlib2d.RFMM(source, dumb_targ, charge=tau, compute_target_potential=True)
            tform = time.time() - st
            print('FMMLIB generation took:               {:0.1f}'.format(tform*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tform/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True)
            self_reference_eval = -0.5*out['source']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB self only eval took:           {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, target, charge=tau, compute_target_potential=True)
            target_reference_eval = -0.5*out['target']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB target only eval took:         {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')
        except:
            print('')
            reference = False

# do my FMM (once first, to compile functions...)
FMM = fmm.FMM(px[:20*N_cutoff], py[:20*N_cutoff], kernel_functions, numba_functions_otf, N_equiv, N_cutoff)
FMM.precompute()
FMM.build_expansions(tau)
_ = FMM.evaluate_to_points(px[:20*N_cutoff], py[:20*N_cutoff], True)

st = time.time()
print('')
FMM = fmm.FMM(px, py, kernel_functions, numba_functions_otf, N_equiv, N_cutoff)
FMM.precompute()
print('pyfmmlib2d precompute took:           {:0.1f}'.format((time.time()-st)*1000))
st = time.time()
FMM.build_expansions(tau)
tt = (time.time()-st)
print('pyfmmlib2d generation took:           {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
st = time.time()
self_fmm_eval = FMM.evaluate_to_points(px, py, True)
tt = (time.time()-st)
print('pyfmmlib2d source eval took:          {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')

st = time.time()
target_fmm_eval = FMM.evaluate_to_points(rx, ry)
tt = (time.time()-st)
print('pyfmmlib2d target eval took:          {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

if reference:
    scale = np.abs(self_reference_eval).max()
    self_err = np.abs(self_fmm_eval - self_reference_eval)/scale
    target_err = np.abs(target_fmm_eval - target_reference_eval)/scale
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))

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


import line_profiler
%load_ext line_profiler
%lprun -f FMM.evaluate_to_points FMM.evaluate_to_points(rx, ry)

%lprun -f FMM.build_expansions FMM.build_expansions(tau)

