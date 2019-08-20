import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import numba
import time
from .tree import Tree
from .misc.mkl_sparse import SpMV_viaMKL
import sys

@numba.njit(parallel=True)
def numba_distribute(ucs, temp, pi, li, li2):
    for i in numba.prange(pi.size):
        if li[i] >= 0:
            ucs[li2[pi[i]]] = temp[li[i]]

@numba.njit(parallel=True)
def numba_add_interactions(doit, ci4, colleagues, xmid, ymid, Local_Solutions, M2Ms, Nequiv, li):
    n = doit.shape[0]
    for i in numba.prange(n):
        if doit[i]:
            dii = ci4[i]
            for j in range(9):
                ci = colleagues[i,j]
                if ci >= 0 and ci != i:
                    xdist = int(np.sign(xmid[ci]-xmid[i]))
                    ydist = int(np.sign(ymid[ci]-ymid[i]))
                    di = li[ci4[ci]]
                    if di >= 0:
                        for k in range(4):
                            for ll in range(Local_Solutions.shape[1]):
                                Local_Solutions[4*dii+k, ll] += \
                                    M2Ms[xdist+1,ydist+1,k*Nequiv+ll,di]

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
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

def prepare_numba_functions_on_the_fly(Kernel_Eval):

    @numba.njit(parallel=True, fastmath=True)
    def numba_upwards_pass(x, y, botind, topind, ns, compute_upwards, xtarg, ytarg, xmid, ymid, tau, ucheck, li):
        n = botind.shape[0]
        ne = xtarg.shape[0]
        for i in numba.prange(n):
            if compute_upwards[i] and li[i] >= 0:
                bi = botind[i]
                ti = topind[i]
                ni = ti-bi
                tx = xtarg + xmid[i]
                ty = ytarg + ymid[i]
                for ks in range(bi, bi+ni):
                    for kt in range(ne):
                        ucheck[li[i],kt] += Kernel_Eval(x[ks], y[ks], tx[kt], ty[kt])*tau[ks]

    @numba.njit(parallel=True, fastmath=True)
    def numba_target_local_expansion_evaluation(xs, ys, inds, locs, large_xs, large_ys, xmids, ymids, Local_Expansions, pot):
        n = xs.size
        for i in numba.prange(n):
            x = xs[i]
            y = ys[i]
            ind = inds[i]
            loc = locs[i]
            tx = x - xmids[ind][loc]
            ty = y - ymids[ind][loc]
            large_x = large_xs[ind]
            large_y = large_ys[ind]
            expansion = Local_Expansions[ind-1][loc] # ind-1 since the 0 one has no local expansions!
            pot[i] = 0.0
            for ks in range(large_x.size):
                pot[i] += Kernel_Eval(large_x[ks], large_y[ks], tx, ty)*expansion[ks]

    @numba.njit(parallel=True, fastmath=True)
    def numba_target_neighbor_evaluation(tx, ty, sx, sy, inds, locs, bot_inds, top_inds, colleagues, tau_ordered, pot, check):
        n = tx.size
        for i in numba.prange(n):
            x = tx[i]
            y = ty[i]
            ind = inds[i]
            loc = locs[i]
            cols = colleagues[ind][loc]
            for j in range(9):
                ci = cols[j]
                if ci >= 0:
                    bind = bot_inds[ind][ci]
                    tind = top_inds[ind][ci]
                    ns = tind - bind
                    if ns > 0 and check:
                        for ks in range(bind, bind+ns):
                            if not (x - sx[ks] == 0 and y - sy[ks] == 0):
                                pot[i] += Kernel_Eval(sx[ks], sy[ks], x, y)*tau_ordered[ks]
                    if ns > 0 and not check:
                        for ks in range(bind, bind+ns):
                            pot[i] += Kernel_Eval(sx[ks], sy[ks], x, y)*tau_ordered[ks]

    return numba_upwards_pass, numba_target_local_expansion_evaluation, numba_target_neighbor_evaluation

def Get_Kernel_Functions(Kernel_Eval):
    @numba.njit(parallel=True, fastmath=True)
    def KF(sx, sy, tx, ty, out):
        ns = sx.shape[0]
        nt = tx.shape[0]
        for i in numba.prange(ns):
            for j in range(nt):
                out[j,i] = Kernel_Eval(sx[i], sy[i], tx[j], ty[j])

    @numba.njit(parallel=True, fastmath=True)
    def KA(sx, sy, tx, ty, tau, out):
        ns = sx.shape[0]
        nt = tx.shape[0]
        for j in numba.prange(nt):
            ja = 0.0
            for i in range(ns):
                ja += Kernel_Eval(sx[i], sy[i], tx[j], ty[j])*tau[i]
            out[j] = ja

    @numba.njit(parallel=True, fastmath=False)                
    def KAS(sx, sy, tau, out):
        ns = sx.shape[0]
        nt = sx.shape[0]
        for j in numba.prange(ns):
            ja = 0.0
            for i in range(ns):
                if i != j:
                    ja += Kernel_Eval(sx[i], sy[i], sx[j], sy[j])*tau[i]
            out[j] = ja

    return KF, KA, KAS

def Kernel_Form(KF, sx, sy, tx=None, ty=None, out=None, mdtype=float):
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
        out = np.empty((nt, ns), dtype=mdtype)
    KF(sx, sy, tx, ty, out)
    if isself:
        np.fill_diagonal(out, 0.0)
    return out

class FMM(object):
    def __init__(self, x, y, kernel_functions, numba_functions, Nequiv=48, Ncutoff=50, iscomplex=False, verbose=False):
        self.x = x
        self.y = y
        self.Nequiv = Nequiv
        self.Ncutoff = Ncutoff
        self.kernel_functions = kernel_functions
        self.numba_functions = numba_functions
        self.dtype = np.complex128 if iscomplex else np.float64
        self.verbose = verbose
        self.print = get_print_function(self.verbose)
    def build_tree(self):
        st = time.time()
        self.tree = Tree(self.x, self.y, self.Ncutoff)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
    def precompute(self):
        if not hasattr(self, 'tree'):
            self.build_tree()
        tree = self.tree
        if not tree.workspace_allocated:
            tree.allocate_workspace(self.Nequiv, dtype=self.dtype)
        st = time.time()
        # get check/equiv surfaces for every level
        Nequiv, Ncutoff = self.Nequiv, self.Ncutoff
        KF, KA, KAS = self.kernel_functions
        theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
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
            equiv_to_check = Kernel_Form(KF, small_xs[ind], small_ys[ind], \
                                                    large_xs[ind], large_ys[ind], mdtype=self.dtype)
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
            Kern = Kernel_Form(KF, collected_equiv_xs, collected_equiv_ys, \
                                                large_xs[ind], large_ys[ind], mdtype=self.dtype)
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
                        M2Lhere[indx,indy] = Kernel_Form(KF, small_xhere, \
                                                small_yhere, small_xs[ind], small_ys[ind], mdtype=self.dtype)
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
                        CM2Lh = np.empty([4*Nequiv, 4*Nequiv], dtype=self.dtype)
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
        self.E2C_LUs = E2C_LUs
        self.M2MC = M2MC
        self.M2LS = M2LS
        self.CM2LS = CM2LS
        self.small_xs = small_xs
        self.small_ys = small_ys
        self.large_xs = large_xs
        self.large_ys = large_ys
        self.small_radii = small_radii
        self.large_radii = large_radii
        self.widths = widths
        et = time.time()
        self.print('....Time for precomputations:   {:0.2f}'.format(1000*(et-st)))
    def build_expansions(self, tau):
        tree, E2C_LUs, M2MC, M2LS, CM2LS, small_xs, small_ys, large_xs, large_ys, small_radii, large_radii, widths = self._get_names()
        (numba_upwards_pass, numba_target_local_expansion_evaluation, \
            numba_target_neighbor_evaluation) = self.numba_functions
        Nequiv, Ncutoff = self.Nequiv, self.Ncutoff
        tau_ordered = tau[tree.ordv]
        self.tau = tau
        self.tau_ordered = tau_ordered
        # upwards pass - start at bottom leaf nodes and build multipoles up
        st = time.time()
        for ind in reversed(range(tree.levels)[1:]):
            Level = tree.Levels[ind]
            u_check_surfaces = Level.Check_Us
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = M2MC[ind].dot(ancestor_level.RSEQD.T).T
                # self.print(temp1.shape, u_check_surfaces.shape, ancestor_level.short_parent_ind.shape, ancestor_level.short_parent_ind.max(), ancestor_level.parent_density_ind.shape, ancestor_level.parent_density_ind.max(), Level.this_density_ind.shape, Level.this_density_ind.max())
                numba_distribute(u_check_surfaces, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            numba_upwards_pass(tree.x, tree.y, Level.bot_ind, Level.top_ind, Level.ns, Level.compute_upwards, large_xs[ind], large_ys[ind], Level.xmid, Level.ymid, tau_ordered, u_check_surfaces, Level.this_density_ind)
            Level.Equiv_Densities[:] = sp.linalg.lu_solve(E2C_LUs[ind], u_check_surfaces.T).T
        et = time.time()
        self.print('....Time for upwards pass:      {:0.2f}'.format(1000*(et-st)))
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
            M2Ms = np.empty([3,3,4*Nequiv,descendant_level.n_sib_has_source], dtype=self.dtype)
            CM2Lh = CM2LS[ind+1]
            for kkx in range(3):
                for kky in range(3):
                    if not (kkx-1 == 0 and kky-1 == 0):
                        CM2Lh[kkx, kky].dot(descendant_level.RSEQD.T, out=M2Ms[kkx, kky])
            ci4 = (Level.children_ind/4).astype(int)
            numba_add_interactions(doit, ci4, Level.colleagues, Level.xmid, Level.ymid, descendant_level.Local_Solutions, M2Ms, Nequiv, descendant_level.parent_density_ind)
        et = time.time()
        self.print('....Time for downwards pass 1:  {:0.2f}'.format(1000*(et-st)))
        # downwards pass 2a - start at top and compute local expansions
        st = time.time()
        for ind in range(1,tree.levels):
            Level = tree.Levels[ind]
            Level.Local_Expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions.T).T
        et = time.time()
        self.print('....Time for downwards pass 2a: {:0.2f}'.format(1000*(et-st)))
    def evaluate_to_point(self, x, y):
        tree, E2C_LUs, M2MC, M2LS, CM2LS, small_xs, small_ys, large_xs, large_ys, small_radii, large_radii, widths = self._get_names()
        KF, KA, KAS = self.kernel_functions
        tau_ordered = self.tau_ordered
        # get level ind, level loc for the point (x, y)
        ind, loc = tree.locate_point(x, y)
        Level = tree.Levels[ind]
        # evaluate appropriate local expansion at (x, y)
        local_expansion = Level.Local_Expansions[loc]
        tx = np.array([x - Level.xmid[loc],])
        ty = np.array([y - Level.ymid[loc],])
        pot = np.array([0.0j,])
        KA(large_xs[ind], large_ys[ind], tx, ty, local_expansion, pot)
        # evaluate interactions from neighbor cells to (x, y)
        colleagues = Level.colleagues[loc]
        for i in range(9):
            ci = colleagues[i]
            if ci >= 0:
                bind = Level.bot_ind[ci]
                tind = Level.top_ind[ci]
                ns = tind - bind
                if ns > 0:
                    cpot = np.array([0.0j,])
                    KA(tree.x[bind:tind], tree.y[bind:tind], np.array([x,]), np.array([y,]), tau_ordered[bind:tind], cpot)
                    pot += cpot
        return pot
    def evaluate_to_points(self, x,  y, check_self=False):
        tree, E2C_LUs, M2MC, M2LS, CM2LS, small_xs, small_ys, large_xs, large_ys, small_radii, large_radii, widths = self._get_names()
        (numba_upwards_pass, numba_target_local_expansion_evaluation, \
            numba_target_neighbor_evaluation) = self.numba_functions
        KF, KA, KAS = self.kernel_functions
        tau_ordered = self.tau_ordered
        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate local expansions
        pot = np.zeros(x.size, dtype=self.dtype)
        Local_Expansions = [Level.Local_Expansions for Level in tree.Levels if hasattr(Level, 'Local_Expansions')]
        bot_inds = [Level.bot_ind for Level in tree.Levels]
        top_inds = [Level.top_ind for Level in tree.Levels]
        colleagues = [Level.colleagues for Level in tree.Levels]
        numba_target_local_expansion_evaluation(x, y, inds, locs, large_xs, large_ys, tree.xmids, tree.ymids, Local_Expansions, pot)
        # evaluate interactions from neighbor cells to (x, y)
        if check_self:
            numba_target_neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, bot_inds, top_inds, colleagues, tau_ordered, pot, True)
        else:
            numba_target_neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, bot_inds, top_inds, colleagues, tau_ordered, pot, False)
        return pot
    def _get_names(self):
        return self.tree, self.E2C_LUs, self.M2MC, self.M2LS, self.CM2LS, self.small_xs, self.small_ys, \
            self.large_xs, self.large_ys, self.small_radii, self.large_radii, self.widths



