import numpy as np
import numexpr as ne
import numba
from scipy.special import struve, y0

# kernel evaluations for interpolater
neach = 10000

def base_gf(x):
    Y = y0(x)
    H = struve(0, x)
    return 0.25*(H-Y)

fit_xb, fit_hb = np.linspace(1.0, 200.0, 10*neach, retstep=True)
fit_yb = base_gf(fit_xb)

cutb = 1.0 + 5*fit_hb
fit_x0, fit_h0 = np.linspace(0.1, 1.0+10*fit_hb, neach, retstep=True)
fit_y0 = base_gf(fit_x0)

cut0 = 0.1 + 5*fit_h0
fit_x1, fit_h1 = np.linspace(0, cut0+10*fit_h0, neach, retstep=True)
fit_y1 = base_gf(fit_x1)

cut1 = 0.001 + 5*fit_h1
fit_x2, fit_h2 = np.linspace(0, cut1+10*fit_h1, neach, retstep=True)
fit_y2 = base_gf(fit_x2)

cut2 = 0.00001 + 5*fit_h2
fit_x3, fit_h3 = np.linspace(0, cut2+10*fit_h2, neach, retstep=True)
fit_y3 = base_gf(fit_x3)

cut3 = 1.0e-7 + 5*fit_h3
fit_x4, fit_h4 = np.linspace(0, cut3+10*fit_h3, neach, retstep=True)
fit_y4 = base_gf(fit_x4)

cut4 = 1.0e-9 + 5*fit_h4
fit_x5, fit_h5 = np.linspace(0, cut4+10*fit_h4, neach, retstep=True)
fit_y5 = base_gf(fit_x5)

cut5 = 1.0e-11 + 5*fit_h5
fit_x6, fit_h6 = np.linspace(0, cut5+10*fit_h5, neach, retstep=True)
fit_y6 = base_gf(fit_x6)

cut6 = 1.0e-13 + 5*fit_h6
fit_x7, fit_h7 = np.linspace(0, cut6+10*fit_h6, neach, retstep=True)
fit_y7 = base_gf(fit_x7)

# 6th order accurate interpolation of smooth f on the interval [a+2*h,b-2*h]
# for f given on [a,a+h,...,b]
# no bounds checking done (on purpose...)
# this will likely segfault for x outside of these bounds
@numba.njit(parallel=False)
def interp(f, xr, a, h):
    xx = xr - a
    ix = int(xx//h)
    ratx = xx/h - (ix+0.5)
    asx = np.empty(6)
    asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
    asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
    asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
    asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
    asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
    asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
    ix -= 2
    fout = 0.0
    for i in range(6):
        fout += f[ix+i]*asx[i]
    return fout

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




