import numpy as np
import numexpr as ne
import numba

@numba.njit("f8(f8,f8,f8,f8)", fastmath=True)
def laplace_eval(sx, sy, tx, ty):
    scale = -0.25/np.pi
    return scale*np.log((tx-sx)**2 + (ty-sy)**2)

@numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])",parallel=True)
def laplace_kernel(sx, sy, tx, ty, shiftx, shifty, charge, pot):
    """
    Numba-jitted Laplace Kernel
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
    scale = -0.25/np.pi
    for i in numba.prange(nt):
        temp = np.zeros(ns)
        for j in range(ns):
            dx = tx[i] + shiftx - sx[j]
            dy = ty[i] + shifty - sy[j]
            temp[j] = dx**2 + dy**2
        for j in range(ns):
            temp[j] = np.log(temp[j])
        for j in range(ns):
            pot[i] += charge[j]*temp[j]*scale
@numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=True)
def laplace_kernel_self(sx, sy, charge, pot):
    """
    Numba-jitted Laplace Kernel
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
    scale = -0.25/np.pi
    for i in numba.prange(ns):
        temp = np.zeros(ns)
        for j in range(ns):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            temp[j] = dx**2 + dy**2
        for j in range(ns):
            temp[j] = np.log(temp[j])
        for j in range(ns):
            if i != j:
                pot[i] += charge[j]*temp[j]*scale


@numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])",parallel=False)
def laplace_kernel_serial(sx, sy, tx, ty, shiftx, shifty, charge, pot):
    """
    Numba-jitted Laplace Kernel
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
    scale = -0.25/np.pi
    for i in range(nt):
        temp = np.zeros(ns)
        for j in range(ns):
            dx = tx[i] + shiftx - sx[j]
            dy = ty[i] + shifty - sy[j]
            temp[j] = dx**2 + dy**2
        for j in range(ns):
            temp[j] = np.log(temp[j])
        for j in range(ns):
            pot[i] += charge[j]*temp[j]*scale
@numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=False)
def laplace_kernel_self_serial(sx, sy, charge, pot):
    """
    Numba-jitted Laplace Kernel
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
    scale = -0.25/np.pi
    for i in range(ns):
        temp = np.zeros(ns)
        for j in range(ns):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            temp[j] = dx**2 + dy**2
        for j in range(ns):
            temp[j] = np.log(temp[j])
        for j in range(ns):
            if i != j:
                pot[i] += charge[j]*temp[j]*scale

def Laplace_Kernel_Form(sx, sy, tx=None, ty=None, out=None):
    """
    Laplace Kernel Formation
    Computes the matrix:
        G_ij
        where G is the Laplace Greens function

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
    scale = -0.5/np.pi
    if out is None:
        out = np.empty([nt, ns], dtype=float)
    ne.evaluate('scale*0.5*log((txt - sx)**2 + (tyt - sy)**2)', out=out)
    if is_self:
        np.fill_diagonal(out, 0.0)
    return out

# @numba.njit("(f8[:],f8[:],f8[:],f8[:],f8,f8,f8[:],f8[:])",parallel=True)
# def _laplace_kernel(sx, sy, tx, ty, shiftx, shifty, charge, pot):
#     """
#     Numba-jitted Laplace Kernel
#     Incoming: charge
#     Outgoing: potential
#     This assumes no overlaps between source and target points
#     Inputs:
#         sx,     intent(in),  float(ns), x-coordinates of source
#         sy,     intent(in),  float(ns), y-coordinates of source
#         tx,     intent(in),  float(nt), x-coordinates of target
#         ty,     intent(in),  float(nt), y-coordinates of target
#         shiftx, intent(in),  float,     target shift in x-direction
#         shifty, intent(in),  float,     target shift in y-direction
#         charge, intent(in),  float(ns), charge at source locations
#         pot,    intent(out), float(nt), potential at target locations
#     ns = number of source points; nt = number of target points
#     all inputs are required
#     """
#     ns = sx.shape[0]
#     nt = tx.shape[0]
#     for i in numba.prange(nt):
#         temp = np.zeros(ns)
#         for j in range(ns):
#             dx = tx[i] + shiftx - sx[j]
#             dy = ty[i] + shifty - sy[j]
#             temp[j] = dx**2 + dy**2
#         for j in range(ns):
#             temp[j] = np.log(temp[j])
#         for j in range(ns):
#             pot[i] += 0.5*charge[j]*temp[j]
# @numba.njit("(f8[:],f8[:],f8[:],f8[:])",parallel=True)
# def _laplace_kernel_self(sx, sy, charge, pot):
#     """
#     Numba-jitted Laplace Kernel
#     Incoming: charge
#     Outgoing: potential
#     This assumes no overlaps between source and target points
#         other than the diagonal term, which is ignored
#     Inputs:
#         sx,     intent(in),  float(ns), x-coordinates of source
#         sy,     intent(in),  float(ns), y-coordinates of source
#         charge, intent(in),  float(ns), charge at source locations
#         pot,    intent(out), float(ns), potential at target locations
#     ns = number of source points
#     all inputs are required
#     """
#     ns = sx.shape[0]
#     for i in numba.prange(ns):
#         temp = np.zeros(ns)
#         for j in range(ns):
#             dx = sx[i] - sx[j]
#             dy = sy[i] - sy[j]
#             temp[j] = dx**2 + dy**2
#         for j in range(ns):
#             temp[j] = np.log(temp[j])
#         for j in range(ns):
#             if i != j:
#                 pot[i] += 0.5*charge[j]*temp[j]
# def Laplace_Kernel_Apply(sx, sy, charge, tx=None, ty=None, tsx=0.0, tsy=0.0, pot=None):
#     is_self = tx is None or ty is None
#     scale = -0.5/np.pi
#     if pot is None:
#         n_out = sx.shape[0] if is_self else tx.shape[0]
#         pot = np.zeros(n_out, dtype=float)
#     else:
#         pot *= 0.0
#     if is_self:
#         _laplace_kernel_self(sx, sy, charge*scale, pot)
#     else:
#         _laplace_kernel(sx, sy, tx, ty, tsx, tsy, charge*scale, pot)
#     return pot
