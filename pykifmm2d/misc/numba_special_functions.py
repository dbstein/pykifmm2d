import numpy as np
import numba

"""
All agorithms from:
Cephes Math Library Release 2.0:  April, 1987
Copyright 1985, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
"""

"""
Chebyshev coefficients for K0(x) + log(x/2) I0(x)
in the interval [0,2].  The odd order coefficients are all
zero; only the even order coefficients are listed.

lim(x->0){ K0(x) + log(x/2) I0(x) } = -EUL.
"""
k0_lt2 = np.array([
    1.37446543561352307156E-16,
    4.25981614279661018399E-14,
    1.03496952576338420167E-11,
    1.90451637722020886025E-9,
    2.53479107902614945675E-7,
    2.28621210311945178607E-5,
    1.26461541144692592338E-3,
    3.59799365153615016266E-2,
    3.44289899924628486886E-1,
    -5.35327393233902768720E-1
], dtype=float)

"""
Chebyshev coefficients for exp(x) sqrt(x) K0(x)
in the inverted interval [2,infinity].

lim(x->inf){ exp(x) sqrt(x) K0(x) } = sqrt(pi/2).
"""
k0_gt2 = np.array([
    5.30043377268626276149E-18,
    -1.64758043015242134646E-17,
    5.21039150503902756861E-17,
    -1.67823109680541210385E-16,
    5.51205597852431940784E-16,
    -1.84859337734377901440E-15,
    6.34007647740507060557E-15,
    -2.22751332699166985548E-14,
    8.03289077536357521100E-14,
    -2.98009692317273043925E-13,
    1.14034058820847496303E-12,
    -4.51459788337394416547E-12,
    1.85594911495471785253E-11,
    -7.95748924447710747776E-11,
    3.57739728140030116597E-10,
    -1.69753450938905987466E-9,
    8.57403401741422608519E-9,
    -4.66048989768794782956E-8,
    2.76681363944501510342E-7,
    -1.83175552271911948767E-6,
    1.39498137188764993662E-5,
    -1.28495495816278026384E-4,
    1.56988388573005337491E-3,
    -3.14481013119645005427E-2,
    2.44030308206595545468E0
], dtype=float)

"""
Chebyshev coefficients for exp(-x) I0(x)
in the interval [0,8].

lim(x->0){ exp(-x) I0(x) } = 1.
"""
i0_lt8 = np.array([
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
], dtype=float)

"""
Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
in the inverted interval [8,infinity].

lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
"""
i0_gt8 = np.array([
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
], dtype=float)

x0 = np.empty((0), dtype=np.float64)
x0.flags.writeable = False
numba.float64(numba.float64, numba.typeof(x0))

@numba.njit(numba.float64(numba.float64, numba.typeof(x0)))
def _numba_chbevl(x, coefs):
    N = coefs.shape[0]
    b0 = coefs[0]
    b1 = 0.0
    for i in range(1,N):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + coefs[i]
    return 0.5*(b0-b2)

@numba.njit("f8(f8)")
def _numba_i0(x):
    """
    computes i0 to 15 digits
    """
    if x < 0:
        x = -x
    if x <= 8.0:
        y = x/2.0 - 2.0
        return np.exp(x)*_numba_chbevl(y, i0_lt8)
    else:
        y = 32.0/x - 2.0
        return np.exp(x)*_numba_chbevl(y, i0_gt8)/np.sqrt(x)

@numba.njit("f8(f8)")
def _numba_k0(x):
    """
    computes k0 to 15 digits
    """
    if x < 0:
        return np.nan
    if x == 0:
        return np.inf
    elif x < 2:
        y = x*x - 2.0
        return _numba_chbevl(y, k0_lt2) - np.log(0.5*x)*_numba_i0(x)
    else:
        z = 8.0/x - 2.0
        return np.exp(-x)*_numba_chbevl(z, k0_gt2)/np.sqrt(x)

# @numba.njit(["f8[:](f8[:])", "f8[:,:](f8[:,:])", "f8[:,:,:](f8[:,:,:])"],parallel=True)
# def numba_k0(x):
#     sh = x.shape
#     N = np.prod(np.array(sh))
#     x = x.ravel()
#     out = np.empty_like(x)
#     for i in numba.prange(N):
#         out[i] = _numba_k0(x[i])
#     return out.reshape(sh)

@numba.vectorize("f8(f8)")
def numba_k0(x):
    return _numba_k0(x)

# @numba.njit("(f8[:,:], i8)", parallel=True)
# def numba_k0_inplace(x, dummy):
#     sh = x.shape
#     for i in numba.prange(sh[0]):
#         for j in range(sh[1]):
#             x[i,j] = _numba_k0(x[i,j])

# @numba.njit(["(f8[:], i8)", "(f8[:,:], i8)", "(f8[:,:,:], i8)"], parallel=True)
# def numba_k0_inplace(x, dummy):
#     sh = x.shape
#     N = np.prod(np.array(sh))
#     x = x.ravel()
#     for i in numba.prange(N):
#         x[i] = _numba_k0(x[i])

