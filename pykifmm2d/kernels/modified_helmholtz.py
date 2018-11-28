import numpy as np
import numexpr as ne
import numba
from ..misc.numba_special_functions import numba_k0, _numba_k0

def generate_modified_helmholtz_functions(k):
    @numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])",parallel=True)
    def modified_helmholtz_kernel(sx, sy, tx, ty, shiftx, shifty, charge, pot):
        ns = sx.shape[0]
        nt = tx.shape[0]
        for i in numba.prange(nt):
            for j in range(ns):
                dx = tx[i] + shiftx - sx[j]
                dy = ty[i] + shifty - sy[j]
                d = np.sqrt(dx**2 + dy**2)
                pot[i] += charge[j]*_numba_k0(k*d)
    @numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=True)
    def modified_helmholtz_kernel_self(sx, sy, charge, pot):
        ns = sx.shape[0]
        scale = -0.25/np.pi
        for i in numba.prange(ns):
            temp = np.zeros(ns)
            for j in range(ns):
                if i != j:
                    dx = sx[i] - sx[j]
                    dy = sy[i] - sy[j]
                    d = np.sqrt(dx**2 + dy**2)
                    pot[i] += charge[j]*_numba_k0(k*d)
    def Modified_Helmholtz_Kernel_Form(sx, sy, tx=None, ty=None):
        is_self = tx is None or ty is None
        if is_self:
            tx = sx
            ty = sy
        ns = sx.shape[0]
        nt = tx.shape[0]
        txt = tx[:,None]
        tyt = ty[:,None]
        G = np.zeros([nt, ns], dtype=float)
        dx = ne.evaluate('txt - sx')
        dy = ne.evaluate('tyt - sy')
        d = ne.evaluate('sqrt(dx**2 + dy**2)')
        G = numba_k0(k*d)
        if is_self:
            np.fill_diagonal(G, 0.0)
        return G
    return Modified_Helmholtz_Kernel_Form, modified_helmholtz_kernel, modified_helmholtz_kernel_self
