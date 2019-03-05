import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import numba
import time
from .tree import Tree
from .misc.mkl_sparse import SpMV_viaMKL

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

def on_the_fly_fmm(x, y, tau, Nequiv, Ncutoff, Kernel_Form, numba_functions, verbose=False):
    """
    On-the-fly KIFMM
        computes sum_{i!=j} G(x_i,x_j) * tau_j
        for greens function G specified in the functions:
            Kernel_Apply and Kernel_Form
    Inputs (all required except Kernel Apply and verbose):
        x,       float(nsource): x-coordinates of sources 
        y,       float(nsource): y-coordinates of sources
        tau,     float(nsource): density
        Nequiv,  int:            number of points used in check/equiv surfaces
        Ncutoff, int:            maximum number of points per leaf node
        Kernel_Apply, function:  Kernel Apply function
        Kernel_Form,  function:  Kernel Form function
        verbose, bool:           enable verbose output
    Outputs:
        potential, float(nsource)

    Notes on inputs:
        Nequiv determines the precision of the solution
            For the Laplace problem, N=64 gives ~machine precision
        Ncutoff determines the balance of work between FMM and local evals
            The best value for this depends on your machine and how efficient
            your Kernel_Apply/Kernel_Form functions are. If your functions are
            very efficient, set this high (like 2000) for best efficiency. If
            your functions are very slow, set this low but larger than Nequiv.
        Kernel_Form:
            This is a function that evaluates a density, with inputs:
                (sx, sy, tau, tx=None, ty=None)
                where sx, sy are source nodes; tx, ty are target nodes
                    and tau is the density
                if tx and ty are not provided, should provide a 'self-eval',
                    i.e. not computing the diagonal terms
        Kernel_Apply:
            This is a function that outputs an evaluation matrix, with inputs:
                (sx, sy, tx=None, ty=None)
                where sx, sy are source nodes; tx, ty are target nodes
                if tx and ty are not provided, should provide a 'self-eval'
                matrix, i.e. with zero on the diagonal
            If this function is not provided, one will be generated using the
                kernel_form function, instead.
        See examples for more information on how to construct the Kernel_Form
            and the Kernel_Apply functions
    """
    my_print = get_print_function(verbose)
    my_print('\nBeginning FMM')

    # build the tree
    st = time.time()
    tree = Tree(x, y, Ncutoff)
    tree_formation_time = (time.time() - st)*1000
    my_print('....Tree formed in:            {:0.1f}'.format(tree_formation_time))

    if tree.levels <= 2:
        # just do a direct evaluation in this case
        solution = np.zeros(tau.shape[0], dtype=float)
        Kernel_Apply(x, y, tau, solution)
    else:
        solution = _on_the_fly_fmm(tree, tau, Nequiv, Kernel_Form, numba_functions, verbose)
    fmm_time = (time.time()-st)*1000
    my_print('FMM completed in               {:0.1f}'.format(fmm_time))
    return solution, tree

def prepare_numba_functions(Kernel_Apply, Kernel_Self_Apply, Kernel_Eval):
    @numba.njit("(f8[:],f8[:],b1[:],i8[:],i8[:],f8[:],i8[:,:],f8[:])", parallel=True)
    def evaluate_neighbor_interactions(x, y, leaf, botind, topind, tau, colleagues, sol):
        n = botind.shape[0]
        for i in numba.prange(n):
            if leaf[i]:
                bind1 = botind[i]
                tind1 = topind[i]
                for j in range(9):
                    ci = colleagues[i,j]
                    if ci >= 0:
                        bind2 = botind[ci]
                        tind2 = topind[ci]
                        if ci == i:
                            Kernel_Self_Apply(x[bind1:tind1], y[bind1:tind1], tau[bind1:tind1], sol[bind1:tind1])
                        else:
                            Kernel_Apply(x[bind2:tind2], y[bind2:tind2], x[bind1:tind1], y[bind1:tind1], 0.0, 0.0, tau[bind2:tind2], sol[bind1:tind1])

    @numba.njit("(f8[:],f8[:],b1[:],i8[:],i8[:],i8[:],i8[:,:],i8,i8[:],i8[:],f8[:])", parallel=True)
    def build_neighbor_interactions(x, y, leaf, ns, botind, topind, colleagues, n_data, iis, jjs, data):
        n = botind.shape[0]
        leaf_vals = np.zeros(n, dtype=np.int64)
        for i in range(n):
            track_val = 0
            if leaf[i]:
                for j in range(9):
                    ci = colleagues[i,j]
                    if ci >= 0:
                        leaf_vals[i] += ns[i]*ns[ci]
        start_vals = np.empty(n, dtype=np.int64)
        start_vals[0] = 0
        for i in range(1,n):
            start_vals[i] = start_vals[i-1] + leaf_vals[i-1]
        for i in numba.prange(n):
            track_val = 0
            if leaf[i]:
                bind1 = botind[i]
                tind1 = topind[i]
                n1 = tind1 - bind1
                for j in range(9):
                    ci = colleagues[i,j]
                    if ci >= 0:
                        if ci == i:
                            for iki, ki in enumerate(range(bind1, tind1)):
                                for ikj, kj in enumerate(range(bind1, tind1)):
                                    if ki != kj:
                                        data[start_vals[i]+track_val+ikj*n1+iki] = Kernel_Eval(x[kj],y[kj],x[ki],y[ki])
                                        iis[start_vals[i]+track_val+ikj*n1+iki] = ki
                                        jjs[start_vals[i]+track_val+ikj*n1+iki] = kj
                            track_val += n1*n1
                        else:
                            bind2 = botind[ci]
                            tind2 = topind[ci]
                            n2 = tind2 - bind2
                            for iki, ki in enumerate(range(bind1, tind1)):
                                for ikj, kj in enumerate(range(bind2, tind2)):
                                    data[start_vals[i]+track_val+ikj*n1+iki] = Kernel_Eval(x[kj],y[kj],x[ki],y[ki])
                                    iis[start_vals[i]+track_val+ikj*n1+iki] = ki
                                    jjs[start_vals[i]+track_val+ikj*n1+iki] = kj
                            track_val += n1*n2

    @numba.njit("(f8[:],f8[:],i8[:],i8[:],f8[:],f8[:],f8[:],f8[:],i8[:],i8[:],f8[:],b1[:],i8)", parallel=True)
    def build_upwards_pass(x, y, botind, topind, xmid, ymid, xring, yring, iis, jjs, data, doit, track_val):
        n = botind.shape[0]
        n1 = xring.shape[0]
        start_vals = np.empty(n, dtype=np.int64)
        start_vals[0] = 0
        for i in range(1,n):
            adder = n1*(topind[i-1]-botind[i-1]) if doit[i-1] else 0
            start_vals[i] = start_vals[i-1] + adder
        for i in numba.prange(n):
            if doit[i]:
                bi = botind[i]
                ti = topind[i]
                n2 = ti - bi
                for ki in range(n1):
                    for ikj, kj in enumerate(range(bi, ti)):
                        data[start_vals[i]+ki*n2+ikj] = Kernel_Eval(x[kj],y[kj],xring[ki]+xmid[i],yring[ki]+ymid[i])
                        iis [start_vals[i]+ki*n2+ikj] = ki + i*n1
                        jjs [start_vals[i]+ki*n2+ikj] = kj
                track_val += n1*n2
        return track_val

    @numba.njit("(f8[:],f8[:],i8[:],i8[:],i8[:],b1[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:,:])",parallel=True)
    def numba_upwards_pass(x, y, botind, topind, ns, compute_upwards, xtarg, ytarg, xmid, ymid, tau, ucheck):
        n = botind.shape[0]
        for i in numba.prange(n):
            if compute_upwards[i] and (ns[i] > 0):
                bi = botind[i]
                ti = topind[i]
                Kernel_Apply(x[bi:ti], y[bi:ti], xtarg, ytarg, xmid[i], ymid[i], tau[bi:ti], ucheck[i])

    @numba.njit("(f8[:],f8[:],i8[:],i8[:],i8[:],b1[:],f8[:],f8[:],f8[:],f8[:],f8[:,:],f8[:])",parallel=True)
    def numba_downwards_pass2(x, y, botind, topind, ns, leaf, xsrc, ysrc, xmid, ymid, local_expansions, sol):
        n = botind.shape[0]
        for i in numba.prange(n):
            if leaf[i] and (ns[i] > 0):
                bi = botind[i]
                ti = topind[i]
                Kernel_Apply(xsrc, ysrc, x[bi:ti], y[bi:ti], -xmid[i], -ymid[i], local_expansions[i], sol[bi:ti])
    return evaluate_neighbor_interactions, build_neighbor_interactions, build_upwards_pass, numba_upwards_pass, numba_downwards_pass2

def _on_the_fly_fmm(tree, tau, Nequiv, Kernel_Form, numba_functions, verbose):
    my_print = get_print_function(verbose)

    (evaluate_neighbor_interactions, build_neighbor_interactions,      \
        build_upwards_pass, numba_upwards_pass, numba_downwards_pass2) \
        = numba_functions

    # allocate workspace in tree
    if not tree.workspace_allocated:
        tree.allocate_workspace(Nequiv)

    st = time.time()
    theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
    # need to reorder tau to match tree order
    tau_ordered = tau[tree.ordv]
    solution_ordered = np.zeros_like(tau)
    # get check/equiv surfaces for every level
    small_xs = []
    small_ys = []
    large_xs = []
    large_ys = []
    small_radii = []
    large_radii = []
    widths = []
    for ind in range(tree.levels):
        Level = tree.Levels[ind]
        width = Level.width
        small_x, small_y, large_x, large_y, small_radius, large_radius = \
                                            get_level_information(width, theta)
        small_xs.append(small_x)
        small_ys.append(small_y)
        large_xs.append(large_x)
        large_ys.append(large_y)
        small_radii.append(small_radius)
        large_radii.append(large_radius)
        widths.append(width)
    # get C2E (check solution to equivalent density) operator for each level
    E2C_LUs = []
    for ind in range(tree.levels):
        equiv_to_check = Kernel_Form(small_xs[ind], small_ys[ind], \
                                                large_xs[ind], large_ys[ind])
        E2C_LUs.append(sp.linalg.lu_factor(equiv_to_check))
    # get Collected Equivalent Coordinates for each level
    M2MC = []
    for ind in range(tree.levels-1):
        collected_equiv_xs = np.concatenate([
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
            ])
        collected_equiv_ys = np.concatenate([
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
            ])
        Kern = Kernel_Form(collected_equiv_xs, collected_equiv_ys, \
                                            large_xs[ind], large_ys[ind])
        M2MC.append(Kern)
    # get all required M2L translations
    M2LS = []
    M2LS.append(None)
    for ind in range(1, tree.levels):
        M2Lhere = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2Lhere[indx, indy] = None
                else:
                    small_xhere = small_xs[ind] + (indx - 3)*widths[ind]
                    small_yhere = small_ys[ind] + (indy - 3)*widths[ind]
                    M2Lhere[indx,indy] = Kernel_Form(small_xhere, \
                                            small_yhere, small_xs[ind], small_ys[ind])
        M2LS.append(M2Lhere)
    # get all Collected M2L translations
    CM2LS = []
    CM2LS.append(None)
    base_shifts_x = np.empty([3,3], dtype=int)
    base_shifts_y = np.empty([3,3], dtype=int)
    for kkx in range(3):
        for kky in range(3):
            base_shifts_x[kkx, kky] = 2*(kkx-1)
            base_shifts_y[kkx, kky] = 2*(kky-1)
    for ind in range(1, tree.levels):
        CM2Lhere = np.empty([3,3], dtype=object)
        M2Lhere = M2LS[ind]
        for kkx in range(3):
            for kky in range(3):
                if not (kkx-1 == 0 and kky-1 == 0):
                    CM2Lh = np.empty([4*Nequiv, 4*Nequiv], dtype=float)
                    base_shift_x = base_shifts_x[kkx, kky]
                    base_shift_y = base_shifts_y[kkx, kky]
                    for ii in range(2):
                        for jj in range(2):        
                            shiftx = base_shift_x - ii + 3
                            shifty = base_shift_y - jj + 3
                            base = 2*ii + jj
                            for iii in range(2):
                                for jjj in range(2):
                                    full_shift_x = shiftx + iii
                                    full_shift_y = shifty + jjj
                                    bb = 2*iii + jjj
                                    if full_shift_x-3 in [-1,0,1] and full_shift_y-3 in [-1,0,1]:
                                        CM2Lh[base*Nequiv:(base+1)*Nequiv,bb*Nequiv:(bb+1)*Nequiv] = 0.0
                                    else:
                                        CM2Lh[base*Nequiv:(base+1)*Nequiv,bb*Nequiv:(bb+1)*Nequiv] = \
                                            M2Lhere[full_shift_x, full_shift_y]
                    CM2Lhere[kkx, kky] = CM2Lh
        CM2LS.append(CM2Lhere)
    et = time.time()
    my_print('....Time for prep work:        {:0.2f}'.format(1000*(et-st)))
    # upwards pass - start at bottom leaf nodes and build multipoles up
    st = time.time()
    for ind in reversed(range(tree.levels)[1:]):
        Level = tree.Levels[ind]
        u_check_surfaces = Level.Check_Us
        # check if there is a level below us, if there is, lift all its expansions
        if ind != tree.levels-1:
            ancestor_level = tree.Levels[ind+1]
        if ind != tree.levels-1:
            ancestor_level = tree.Levels[ind+1]
            temp1 = M2MC[ind].dot(ancestor_level.RSEQD.T).T
            numba_distribute(u_check_surfaces, temp1, ancestor_level.short_parent_ind, int(ancestor_level.n_node/4))
        numba_upwards_pass(tree.x, tree.y, Level.bot_ind, Level.top_ind, Level.ns, Level.compute_upwards, large_xs[ind], large_ys[ind], Level.xmid, Level.ymid, tau_ordered, u_check_surfaces)
        Level.Equiv_Densities[:] = sp.linalg.lu_solve(E2C_LUs[ind], u_check_surfaces.T).T
    et = time.time()
    my_print('....Time for upwards pass:     {:0.2f}'.format(1000*(et-st)))
    # downwards pass 1 - start at top and work down to build up local expansions
    st = time.time()
    for ind in range(1, tree.levels-1):
        # first move local expansions downward
        Level = tree.Levels[ind]
        descendant_level = tree.Levels[ind+1]
        doit = np.logical_and(np.logical_or(Level.not_leaf, Level.Xlist), np.logical_not(Level.fake_leaf))
        local_expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions[doit].T).T
        local_solutions = M2MC[ind].T.dot(local_expansions.T).T
        sorter = np.argsort(Level.children_ind[doit])
        local_solutions = local_solutions[sorter]
        # now we have not leaves in the descendant_level.Local_Solutions...
        descendant_level.Local_Solutions[:] = local_solutions.reshape(descendant_level.Local_Solutions.shape)
        # compute all possible interactions
        M2Ms = np.empty([3,3,doit.sum(),4*Nequiv], dtype=float)
        CM2Lh = CM2LS[ind+1]
        for kkx in range(3):
            for kky in range(3):
                if not (kkx-1 == 0 and kky-1 == 0):
                    M2Ms[kkx, kky, :, :] = CM2Lh[kkx, kky].dot(descendant_level.RSEQD.T).T
        ci4 = (Level.children_ind/4).astype(int)
        numba_add_interactions(doit, ci4, Level.colleagues, Level.xmid, Level.ymid, descendant_level.Local_Solutions, M2Ms, Nequiv)
    et = time.time()
    my_print('....Time for downwards pass 1: {:0.2f}'.format(1000*(et-st)))
    # downwards pass 2 - start at top and evaluate local expansions
    st = time.time()
    for ind in range(1,tree.levels):
        Level = tree.Levels[ind]
        local_expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions.T).T
        numba_downwards_pass2(tree.x, tree.y, Level.bot_ind, Level.top_ind, Level.ns, Level.leaf, large_xs[ind], large_ys[ind], Level.xmid, Level.ymid, local_expansions, solution_ordered)
    et = time.time()
    my_print('....Time for downwards pass 2: {:0.2f}'.format(1000*(et-st)))
    solution_save = solution_ordered.copy()
    # downwards pass 3 - start at top and evaluate neighbor interactions
    st = time.time()
    for ind in range(1,tree.levels):
        Level = tree.Levels[ind]
        evaluate_neighbor_interactions(tree.x, tree.y, Level.leaf, Level.bot_ind, Level.top_ind, tau_ordered, Level.colleagues, solution_ordered)
    et = time.time()
    my_print('....Time for downwards pass 3: {:0.2f}'.format(1000*(et-st)))
    # deorder the solution
    desorter = np.argsort(tree.ordv)
    return solution_ordered[desorter]

@numba.njit("(b1[:],i8[:],i8[:,:],f8[:],f8[:],f8[:,:],f8[:,:,:,:],i8)",parallel=True)
def numba_add_interactions(doit, ci4, colleagues, xmid, ymid, Local_Solutions, M2Ms, Nequiv):
    n = doit.shape[0]
    for i in numba.prange(n):
        if doit[i]:
            dii = ci4[i]
            for j in range(9):
                ci = colleagues[i,j]
                if ci >= 0 and ci != i:
                    xdist = int(np.sign(xmid[ci]-xmid[i]))
                    ydist = int(np.sign(ymid[ci]-ymid[i]))
                    di = ci4[ci]
                    for k in range(4):
                        for ll in range(Local_Solutions.shape[1]):
                            Local_Solutions[4*dii+k, ll] += \
                                M2Ms[xdist+1,ydist+1,di,k*Nequiv+ll]

class FMM_Plan(object):
    def __init__(self, tree, theta, large_xs, large_ys, E2C_LUs, M2MC, M2LS, CM2LS, neighbor_mats, upwards_mats, downwards_mats, numba_functions, verbose):
        self.tree = tree
        self.theta = theta
        self.large_xs = large_xs
        self.large_ys = large_ys
        self.E2C_LUs = E2C_LUs
        self.M2MC = M2MC
        self.M2LS = M2LS
        self.CM2LS = CM2LS
        self.neighbor_mats = neighbor_mats
        self.upwards_mats = upwards_mats
        self.downwards_mats = downwards_mats
        self.numba_functions = numba_functions
        self.verbose = verbose
    def extract(self):
        return self.tree, self.theta, self.large_xs, self.large_ys, self.E2C_LUs, self.M2MC, self.M2LS, self.CM2LS, self.neighbor_mats, self.upwards_mats, self.downwards_mats, self.numba_functions, self.verbose

def fmm_planner(x, y, Nequiv, Ncutoff, Kernel_Form, numba_functions, verbose=False):
    my_print = get_print_function(verbose)
    my_print('\nPlanning FMM')

    (evaluate_neighbor_interactions, build_neighbor_interactions,      \
        build_upwards_pass, numba_upwards_pass, numba_downwards_pass2) \
        = numba_functions

    # building a tree
    st = time.time()
    tree = Tree(x, y, Ncutoff)
    tree_formation_time = (time.time() - st)*1000
    my_print('....Tree formed in:              {:0.1f}'.format(tree_formation_time))

    # allocate workspace in tree
    if not tree.workspace_allocated:
        tree.allocate_workspace(Nequiv)

    st = time.time()
    theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
    # get check/equiv surfaces for every level
    small_xs = []
    small_ys = []
    large_xs = []
    large_ys = []
    small_radii = []
    large_radii = []
    widths = []
    for ind in range(tree.levels):
        Level = tree.Levels[ind]
        width = Level.width
        small_x, small_y, large_x, large_y, small_radius, large_radius = \
                                            get_level_information(width, theta)
        small_xs.append(small_x)
        small_ys.append(small_y)
        large_xs.append(large_x)
        large_ys.append(large_y)
        small_radii.append(small_radius)
        large_radii.append(large_radius)
        widths.append(width)
    # get C2E (check solution to equivalent density) operator for each level
    E2C_LUs = []
    for ind in range(tree.levels):
        equiv_to_check = Kernel_Form(small_xs[ind], small_ys[ind], \
                                                large_xs[ind], large_ys[ind])
        E2C_LUs.append(sp.linalg.lu_factor(equiv_to_check))
    # get Collected Equivalent Coordinates for each level
    M2MC = []
    for ind in range(tree.levels-1):
        collected_equiv_xs = np.concatenate([
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
            ])
        collected_equiv_ys = np.concatenate([
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
            ])
        Kern = Kernel_Form(collected_equiv_xs, collected_equiv_ys, \
                                            large_xs[ind], large_ys[ind])
        M2MC.append(Kern)
    # get all required M2L translations
    M2LS = []
    M2LS.append(None)
    for ind in range(1, tree.levels):
        M2Lhere = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2Lhere[indx, indy] = None
                else:
                    small_xhere = small_xs[ind] + (indx - 3)*widths[ind]
                    small_yhere = small_ys[ind] + (indy - 3)*widths[ind]
                    M2Lhere[indx,indy] = Kernel_Form(small_xhere, \
                                            small_yhere, small_xs[ind], small_ys[ind])
        M2LS.append(M2Lhere)
    # get all Collected M2L translations
    CM2LS = []
    CM2LS.append(None)
    base_shifts_x = np.empty([3,3], dtype=int)
    base_shifts_y = np.empty([3,3], dtype=int)
    for kkx in range(3):
        for kky in range(3):
            base_shifts_x[kkx, kky] = 2*(kkx-1)
            base_shifts_y[kkx, kky] = 2*(kky-1)
    for ind in range(1, tree.levels):
        CM2Lhere = np.empty([3,3], dtype=object)
        M2Lhere = M2LS[ind]
        for kkx in range(3):
            for kky in range(3):
                if not (kkx-1 == 0 and kky-1 == 0):
                    CM2Lh = np.empty([4*Nequiv, 4*Nequiv], dtype=float)
                    base_shift_x = base_shifts_x[kkx, kky]
                    base_shift_y = base_shifts_y[kkx, kky]
                    for ii in range(2):
                        for jj in range(2):        
                            shiftx = base_shift_x - ii + 3
                            shifty = base_shift_y - jj + 3
                            base = 2*ii + jj
                            for iii in range(2):
                                for jjj in range(2):
                                    full_shift_x = shiftx + iii
                                    full_shift_y = shifty + jjj
                                    bb = 2*iii + jjj
                                    if full_shift_x-3 in [-1,0,1] and full_shift_y-3 in [-1,0,1]:
                                        CM2Lh[base*Nequiv:(base+1)*Nequiv,bb*Nequiv:(bb+1)*Nequiv] = 0.0
                                    else:
                                        CM2Lh[base*Nequiv:(base+1)*Nequiv,bb*Nequiv:(bb+1)*Nequiv] = \
                                            M2Lhere[full_shift_x, full_shift_y]
                    CM2Lhere[kkx, kky] = CM2Lh.T
        CM2LS.append(CM2Lhere)
    et = time.time()
    my_print('....Time for basic work:         {:0.2f}'.format(1000*(et-st)))

    # generate sparse matrix for neighbor interactions for each level
    st = time.time()
    memory = np.empty([4*Ncutoff,4*Ncutoff], dtype=float)
    base_ranges = np.arange(4*Ncutoff)
    for iL, Level in enumerate(tree.Levels):
        n_data = numba_get_neighbor_length(Level.leaf, Level.ns, Level.colleagues)
        iis = np.zeros(n_data, dtype=int)
        jjs = np.zeros(n_data, dtype=int)
        data = np.zeros(n_data, dtype=float)
        build_neighbor_interactions(tree.x, tree.y, Level.leaf, Level.ns,
            Level.bot_ind, Level.top_ind, Level.colleagues, n_data, iis, jjs, data)
        level_matrix = sp.sparse.coo_matrix((data,(iis,jjs)),shape=[tree.x.shape[0],tree.x.shape[0]])
        if iL == 0:
            neighbor_mat = level_matrix.tocsr()
        else:
            neighbor_mat += level_matrix.tocsr()
    neighbor_mat = neighbor_mat.tocsr()
        # neighbor_mats.append(level_matrix.tocsr())
    # neighbor_mat = neighbor_mats[0]
    # for ind in range(1,tree.levels):
        # neighbor_mat += neighbor_mats[ind]
    et = time.time()
    my_print('....Time to make neighbor mats   {:0.2f}'.format(1000*(et-st)))

    # generate sparse matrix for upwards pass for each level
    st = time.time()
    upwards_mats = []
    for ind, Level in enumerate(tree.Levels):
        iis =  np.empty(Level.n_node*Ncutoff*Nequiv, dtype=int)
        jjs =  np.empty(Level.n_node*Ncutoff*Nequiv, dtype=int)
        data = np.empty(Level.n_node*Ncutoff*Nequiv, dtype=float)
        track_val = 0
        doit = np.logical_and(Level.compute_upwards, Level.ns>0)
        track_val = build_upwards_pass(tree.x, tree.y, Level.bot_ind, Level.top_ind, \
            Level.xmid, Level.ymid, large_xs[ind], large_ys[ind], iis, jjs, \
            data, doit, track_val)              
        iis = iis[:track_val]
        jjs = jjs[:track_val]
        data = data[:track_val]
        level_matrix = sp.sparse.coo_matrix((data,(iis,jjs)),shape=[Nequiv*Level.n_node,tree.x.shape[0]])
        upwards_mats.append(level_matrix.tocsr())
    et = time.time()
    my_print('....Time to make upwards mats    {:0.2f}'.format(1000*(et-st)))

    # generate sparse matrix for downwards pass for each level
    st = time.time()
    downwards_mats = []
    for ind, Level in enumerate(tree.Levels):
        iis =  np.empty(Level.n_node*Ncutoff*Nequiv, dtype=int)
        jjs =  np.empty(Level.n_node*Ncutoff*Nequiv, dtype=int)
        data = np.empty(Level.n_node*Ncutoff*Nequiv, dtype=float)
        track_val = 0
        doit = np.logical_and(Level.leaf, Level.ns>0)
        track_val = build_upwards_pass(tree.x, tree.y, Level.bot_ind, Level.top_ind, \
            Level.xmid, Level.ymid, large_xs[ind], large_ys[ind], iis, jjs, \
            data, doit, track_val)    
        iis = iis[:track_val]
        jjs = jjs[:track_val]
        data = data[:track_val]
        level_matrix = sp.sparse.coo_matrix((data,(iis,jjs)),shape=[Nequiv*Level.n_node,tree.x.shape[0]])
        downwards_mats.append(level_matrix.T.tocsr())
        # downwards_mats.append(upwards_mats[ind].T.tocsr())
    et = time.time()
    my_print('....Time to make downwards mats  {:0.2f}'.format(1000*(et-st)))

    fmm_plan = FMM_Plan(tree, theta, large_xs, large_ys, E2C_LUs, M2MC, M2LS, CM2LS, neighbor_mat, upwards_mats, downwards_mats, numba_functions, verbose)
    return fmm_plan

@numba.njit("i8(b1[:],i8[:],i8[:,:])",parallel=False)
def numba_get_neighbor_length(leaf, ns, colleagues):
    n = 0
    for i in range(leaf.shape[0]):
        if leaf[i] and (ns[i] > 0):
            for j in range(9):
                ci = colleagues[i,j]
                if ci >= 0:
                    n += ns[ci]*ns[i]
    return n

def set_matval(xx, xn, ti):
    nn = xn.shape[0]
    try:
        xx[ti:ti+nn] = xn
    except:
        xxn = np.empty(xx.shape[0]*2, dtype=xx.dtype)
        xxn[:ti] = xx[:ti]
        xxn[ti:ti+nn] = xn
        xx = xxn
    return xx

@numba.njit("(f8[:,:],f8[:,:],i8[:],i8)",parallel=True)
def numba_distribute(ucs, temp, pi, n):
    for i in numba.prange(n):
        ucs[pi[i]] = temp[i]

def planned_fmm(fmm_plan, tau):
    tree, theta, large_xs, large_ys, E2C_LUs, M2MC, M2LS, CM2LS, neighbor_mats, upwards_mats, downwards_mats, numba_functions, verbose \
        = fmm_plan.extract()
    Nequiv = theta.shape[0]

    my_print = get_print_function(verbose)
    (evaluate_neighbor_interactions, build_neighbor_interactions,      \
        build_upwards_pass, numba_upwards_pass, numba_downwards_pass2) \
        = numba_functions

    my_print('Executing FMM')

    tau_ordered = tau[tree.ordv]
    solution_ordered = np.zeros_like(tau)
    # upwards pass - start at bottom leaf nodes and build multipoles up
    st = time.time()
    mat_time = 0
    lu_time = 0
    for ind in reversed(range(tree.levels)[1:]):
        Level = tree.Levels[ind]
        stt = time.time()
        u_check_surfaces = SpMV_viaMKL(upwards_mats[ind], tau_ordered).reshape([Level.n_node, Nequiv])
        mat_time += time.time() - stt
        if ind != tree.levels-1:
            ancestor_level = tree.Levels[ind+1]
            temp1 = M2MC[ind].dot(ancestor_level.RSEQD.T).T
            numba_distribute(u_check_surfaces, temp1, ancestor_level.short_parent_ind, int(ancestor_level.n_node/4))
        stt = time.time()
        Level.Equiv_Densities[:] = sp.linalg.lu_solve(E2C_LUs[ind], u_check_surfaces.T).T
        lu_time += time.time() - stt
    et = time.time()
    my_print('....Time for upwards pass:       {:0.2f}'.format(1000*(et-st)))
    # my_print('....Time for matvecs:            {:0.2f}'.format(1000*mat_time))
    # my_print('....Time for lus:                {:0.2f}'.format(1000*lu_time))
    # downwards pass 1 - start at top and work down to build up local expansions
    st = time.time()
    for ind in range(1, tree.levels-1):
        # first move local expansions downward
        Level = tree.Levels[ind]
        descendant_level = tree.Levels[ind+1]
        doit = np.logical_and(np.logical_or(Level.not_leaf, Level.Xlist), np.logical_not(Level.fake_leaf))
        local_expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions[doit].T).T
        local_solutions = M2MC[ind].T.dot(local_expansions.T).T
        sorter = np.argsort(Level.children_ind[doit])
        local_solutions = local_solutions[sorter]
        # now we have not leaves in the descendant_level.Local_Solutions...
        descendant_level.Local_Solutions[:] = local_solutions.reshape(descendant_level.Local_Solutions.shape)
        # compute all possible interactions
        # do we actually need to do all these?  probably not...
        M2Ms = np.empty([3,3,doit.sum(),4*Nequiv], dtype=float)
        CM2Lh = CM2LS[ind+1]
        for kkx in range(3):
            for kky in range(3):
                if not (kkx-1 == 0 and kky-1 == 0):
                    np.dot(descendant_level.RSEQD, CM2Lh[kkx, kky], out=M2Ms[kkx, kky])
        ci4 = (Level.children_ind/4).astype(int)
        numba_add_interactions(doit, ci4, Level.colleagues, Level.xmid, Level.ymid, descendant_level.Local_Solutions, M2Ms, Nequiv)
    et = time.time()
    my_print('....Time for downwards pass 1:   {:0.2f}'.format(1000*(et-st)))
    # downwards pass 2 - start at top and evaluate local expansions
    st = time.time()
    for ind in range(1,tree.levels):
        Level = tree.Levels[ind]
        local_expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions.T).T
        solution_ordered += SpMV_viaMKL(downwards_mats[ind], local_expansions.ravel())
    et = time.time()
    my_print('....Time for downwards pass 2:   {:0.2f}'.format(1000*(et-st)))
    solution_save = solution_ordered.copy()
    # downwards pass 3 - start at top and evaluate neighbor interactions
    st = time.time()
    solution_ordered += SpMV_viaMKL(neighbor_mats, tau_ordered)
    et = time.time()
    my_print('....Time for downwards pass 3:   {:0.2f}'.format(1000*(et-st)))
    # deorder the solution
    desorter = np.argsort(tree.ordv)
    return solution_ordered[desorter]


