import numpy as np
import numexpr as ne
import numba
from scipy.special import struve, y0

@numba.njit("f8(f8)", parallel=False)
def naomi_eval(x):
    if x < cut6:
        out = interp(fit_y7, x, 0.0, fit_h7)
    elif x < cut5:
        out = interp(fit_y6, x, 0.0, fit_h6)
    elif x < cut4:
        out = interp(fit_y5, x, 0.0, fit_h5)    
    elif x < cut3:
        out = interp(fit_y4, x, 0.0, fit_h4)
    elif x < cut2:
        out = interp(fit_y3, x, 0.0, fit_h3)
    elif x < cut1:
        out = interp(fit_y2, x, 0.0, fit_h2)
    elif x < cut0:
        out = interp(fit_y1, x, 0.0, fit_h1)
    elif x < cutb:
        out = interp(fit_y0, x, 0.1, fit_h0)
    else:
        out = interp(fit_yb, x, 1.0, fit_hb)
    return out

@numba.njit()
def hypot(x, y):
    return np.sqrt(x**2 + y**2)

@numba.njit("f8(f8,f8,f8,f8)")
def naomi_kernel_eval(sx, sy, tx, ty):
    return naomi_eval(hypot(tx-sx,ty-sy))

@numba.njit("(f8[:],f8[:])", parallel=True)
def naomi_multi_eval1(x,out):
    m1 = x.shape[0]
    for i in numba.prange(m1):
        out[i] = naomi_eval(x[i])
@numba.njit("(f8[:,:],f8[:,:])", parallel=True)
def naomi_multi_eval2(x,out):
    m1 = x.shape[0]
    m2 = x.shape[1]
    for i in numba.prange(m1):
        for j in range(m2):
            out[i,j] = naomi_eval(x[i,j])

@numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])", parallel=True)
def naomi_kernel(sx, sy, tx, ty, shiftx, shifty, charge, pot):
    """
    Numba-jitted Naomi Kernel
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
            r = hypot(tx[i] + shiftx - sx[j], ty[i] + shifty - sy[j])
            pot[i] += naomi_eval(r)*charge[j]
@numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=True)
def naomi_kernel_self(sx, sy, charge, pot):
    """
    Numba-jitted Naomi Kernel
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
                r = hypot(sx[i] - sx[j], sy[i] - sy[j])
                pot[i] += naomi_eval(r)*charge[j]
@numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])")
def naomi_kernel_serial(sx, sy, tx, ty, shiftx, shifty, charge, pot):
    """
    Numba-jitted Naomi Kernel
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
    for i in range(nt):
        for j in range(ns):
            r = hypot(tx[i] + shiftx - sx[j], ty[i] + shifty - sy[j])
            pot[i] += naomi_eval(r)*charge[j]
@numba.njit("(f8[:],f8[:],f8[:],f8[:])")
def naomi_kernel_self_serial(sx, sy, charge, pot):
    """
    Numba-jitted Naomi Kernel
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
    for i in range(ns):
        for j in range(ns):
            if i != j:
                r = hypot(sx[i] - sx[j], sy[i] - sy[j])
                pot[i] += naomi_eval(r)*charge[j]
def naomi_kernel_form(sx, sy, tx=None, ty=None, out=None):
    """
    Naomi Kernel Formation

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
    ns = sx.shape[0]
    nt = tx.shape[0]
    txt = tx[:,None]
    tyt = ty[:,None]
    r = ne.evaluate('sqrt((txt - sx)**2 + (tyt - sy)**2)')
    if out is None:
        out = np.empty_like(r)
    naomi_multi_eval2(r, out)
    if is_self:
        np.fill_diagonal(out, 0.0)
    return out




