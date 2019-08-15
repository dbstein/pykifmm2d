import pykifmm2d
import pykifmm2d.svd_fmm
import numpy as np
import numba
import time
import matplotlib as mpl
mpl.use('TkAgg')
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
Prepare_Functions_OTF  = pykifmm2d.fmm.prepare_numba_functions_on_the_fly
Prepare_Functions_PLAN = pykifmm2d.fmm.prepare_numba_functions_planned
Prepare_K_Functions    = pykifmm2d.fmm.Get_Kernel_Functions

# Laplace Kernel
@numba.njit("f8(f8,f8,f8,f8)", fastmath=True)
def Laplace_Kernel_Eval(sx, sy, tx, ty):
    scale = -0.25/np.pi
    return scale*np.log((tx-sx)**2 + (ty-sy)**2)

# associated kernel evaluation functions
kernel_functions = Prepare_K_Functions(Laplace_Kernel_Eval)
(KF, KA, KAS) = kernel_functions
# jit compile internal numba functions
numba_functions_otf  = Prepare_Functions_OTF (Laplace_Kernel_Eval)
numba_functions_plan = Prepare_Functions_PLAN(Laplace_Kernel_Eval)

N_total = 1000*100
test = 'uniform' # clustered or circle or uniform

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

print('\nLaplace Kernel Direct vs. FMM demonstration with', N_total, 'points.')

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
            out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True)
            et = time.time()
            time_fmmlib_eval = (et-st)*1000
            reference_eval = -out['source']['u']/(2*np.pi)
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

















if False:

    import line_profiler
    %load_ext line_profiler
    from pykifmm2d.svd_fmm import _on_the_fly_fmm

    %lprun -f _on_the_fly_fmm pykifmm2d.svd_fmm.on_the_fly_fmm(px, py, tau, N_equiv, N_cutoff, \
                        kernel_functions, numba_functions_otf, verbose=True)

    verbose = True
    numba_functions = numba_functions_otf
    Nequiv = N_equiv
    import numexpr as ne
    cacheit = False
    def get_level_information(node_width, theta):
        # get information for this level
        dd = 0.01
        r1 = 0.5*node_width*(np.sqrt(2)+dd)
        r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
        small_surface_x_base = r1*np.cos(theta)
        small_surface_y_base = r1*np.sin(theta)
        large_surface_x_base = r2*np.cos(theta)
        large_surface_y_base = r2*np.sin(theta)
        return small_surface_x_base, small_surface_y_base, large_surface_x_base, \
                    large_surface_y_base, r1, r2
    def fake_print(*args, **kwargs):
        pass
    def get_print_function(verbose):
        return print if verbose else fake_print
    def Kernel_Form(KF, sx, sy, tx=None, ty=None, out=None):
        if tx is None or ty is None:
            tx = sx
            ty = sy
            isself = True
        else:
            if sx is tx and sy is ty:
                isself = True
            else:
                isself = False
        ns = sx.shape[0]
        nt = tx.shape[0]
        if out is None:
            out = np.empty((nt, ns))
        KF(sx, sy, tx, ty, out)
        if isself:
            np.fill_diagonal(out, 0.0)
        return out
    def Kernel_Apply(KA, KAS, sx, sy, tau, tx=None, ty=None, out=None):
        if tx is None or ty is None:
            tx = sx
            ty = sy
            isself = True
        else:
            if sx is tx and sy is ty:
                isself = True
            else:
                isself = False
        ns = sx.shape[0]
        nt = tx.shape[0]
        if out is None:
            out = np.empty(nt)
        if isself:
            KAS(sx, sy, tau, out)
        else:
            KA(sx, sy, tx, ty, tau, out)
        return out
    def fft_solve(avhi, B):
        bvh = np.fft.rfft(B, axis=0)
        ne.evaluate('bvh*avhi', out=bvh)
        return np.fft.irfft(bvh, axis=0)



