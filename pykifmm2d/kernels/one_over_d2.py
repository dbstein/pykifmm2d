import numpy as np
import numexpr as ne
import numba

@numba.njit("f8(f8,f8,f8,f8)")
def ood2_eval(sx, sy, tx, ty):
    d2 = (tx-sx)**2 + (ty-sy)**2
    return 1.0/d2

@numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])",parallel=True)
def ood2_kernel(sx, sy, tx, ty, shiftx, shifty, charge, pot):
    """
    Numba-jitted Biharmonic Kernel
    Incoming: charge
    Outgoing: potential
    This assumes no overlaps between source and target points
    Inputs:
        sx,     intent(in),  float(ns), x-coordinates of source
        sy,     intent(in),  float(ns), y-coordinates of source
        tx,     intent(in),  float(nt), x-coordinates of target
        ty,     intent(in),  float(nt), y-coordinates of target
        shiftx, intent(in),  float,     target shift in x-direction
        shifty, intent(in),  float,     target shift in y-direction
        charge, intent(in),  float(ns), charge at source locations
        pot,    intent(out), float(nt), potential at target locations
    ns = number of source points; nt = number of target points
    all inputs are required
    """
    ns = sx.shape[0]
    nt = tx.shape[0]
    for i in numba.prange(nt):
        for j in range(ns):
            dx = tx[i] + shiftx - sx[j]
            dy = ty[i] + shifty - sy[j]
            d2 = dx**2 + dy**2
            pot[i] += charge[j]/d2

@numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=True)
def ood2_kernel_self(sx, sy, charge, pot):
    """
    Numba-jitted Biharmonic Kernel
    Incoming: charge
    Outgoing: potential
    This assumes no overlaps between source and target points
        other than the diagonal term, which is ignored
    Inputs:
        sx,     intent(in),  float(ns), x-coordinates of source
        sy,     intent(in),  float(ns), y-coordinates of source
        charge, intent(in),  float(ns), charge at source locations
        pot,    intent(out), float(ns), potential at target locations
    ns = number of source points
    all inputs are required
    """
    ns = sx.shape[0]
    for i in numba.prange(ns):
        for j in range(ns):
            if i != j:
                dx = sx[i] - sx[j]
                dy = sy[i] - sy[j]
                d2 = dx**2 + dy**2
                pot[i] += charge[j]/d2

def ood2_Kernel_Form(sx, sy, tx=None, ty=None, out=None):
    """
    Biharmonic Kernel Formation
    Computes the matrix:
        G_ij
        where G is the Biharmonic Greens function

    Parameters:
        sx,   required, float(ns),  source x-coordinates
        sy,   required, float(ns),  source y-coordinates
        tx,   optional, float(ns),  source x-coordinates
        ty,   optional, float(ns),  source y-coordinates

    If given, this function assumes target and source points are not coincident
    """
    is_self = tx is None or ty is None
    if is_self:
        tx = sx
        ty = sy
    txt = tx[:,None]
    tyt = ty[:,None]
    if out is None:
        out = np.empty([tx.shape[0], sx.shape[0]], dtype=float)
    ne.evaluate('1.0/((txt - sx)**2 + (tyt - sy)**2)', out=out)
    if is_self:
        np.fill_diagonal(out, 0.0)
    return out
