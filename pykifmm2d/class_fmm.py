import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import numba
import time
from .tree import Tree
from .misc.mkl_sparse import SpMV_viaMKL

cacheit = False

@numba.njit(parallel=True, cache=cacheit)
def numba_distribute(ucs, temp, pi, n):
    for i in numba.prange(n):
        ucs[pi[i]] = temp[i]

@numba.njit(parallel=True, cache=cacheit)
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

def prepare_numba_functions_on_the_fly(Kernel_Eval):
    @numba.njit(parallel=True, fastmath=True)
    def evaluate_neighbor_interactions(x, y, leaf, botind, topind, tau, colleagues, sol):
        n = botind.shape[0]
        # loop over all nodes in this level
        for i in numba.prange(n):
            # if it's a leaf, we add neighbor interactions
            if leaf[i]:
                bind1 = botind[i]
                tind1 = topind[i]
                nt = tind1 - bind1 # number of targets in this leaf
                if nt > 0:
                    tx = np.empty(nt)
                    ty = np.empty(nt)
                    soli = np.zeros(nt, dtype=np.complex128)
                    for ll in range(nt):
                        tx[ll] = x[bind1 + ll]
                        ty[ll] = y[bind1 + ll]
                    # loop through all 9 possible colleagues
                    for j in range(9):
                        ci = colleagues[i,j]
                        if ci >= 0: # if ci < 0 there's not really a colleague
                            bind2 = botind[ci]
                            tind2 = topind[ci]
                            ns = tind2 - bind2
                            # number of source points in colleague
                            if ns > 0:
                                sx = np.empty(ns)
                                sy = np.empty(ns)
                                taui = np.empty(ns, dtype=np.complex128)
                                for ll in range(ns):
                                    sx[ll] = x[bind2 + ll]
                                    sy[ll] = y[bind2 + ll]
                                    taui[ll] = tau[bind2 + ll]
                                for ks in range(ns):
                                    for kt in range(nt):
                                        if i != ci or ks != kt:
                                            soli[kt] += taui[ks]*Kernel_Eval(sx[ks], sy[ks], tx[kt], ty[kt])
                    for kt in range(nt):
                        sol[kt + bind1] += soli[kt]

    @numba.njit(parallel=True, fastmath=True)
    def numba_upwards_pass(x, y, botind, topind, ns, compute_upwards, xtarg, ytarg, xmid, ymid, tau, ucheck):
        n = botind.shape[0]
        ne = xtarg.shape[0]
        for i in numba.prange(n):
            if compute_upwards[i] and (ns[i] > 0):
                bi = botind[i]
                ti = topind[i]
                ni = ti-bi
                sx = np.empty(ni)
                sy = np.empty(ni)
                taus = np.empty(ni, dtype=np.complex128)
                uch = np.zeros(ne, dtype=np.complex128)
                for ll in range(ni):
                    sx[ll] = x[bi+ll]
                    sy[ll] = y[bi+ll]
                    taus[ll] = tau[bi+ll]
                tx = xtarg + xmid[i]
                ty = ytarg + ymid[i]
                for ks in range(ni):
                    for kt in range(ne):
                        uch[kt] += Kernel_Eval(sx[ks], sy[ks], tx[kt], ty[kt])*taus[ks]
                for kt in range(ne):
                    ucheck[i,kt] += uch[kt]

    @numba.njit(parallel=True, fastmath=True)
    def numba_downwards_pass2(x, y, botind, topind, ns, leaf, xsrc, ysrc, xmid, ymid, local_expansions, sol):
        n = botind.shape[0]
        ne = xsrc.shape[0]
        sx = np.empty(ne)
        sy = np.empty(ne)
        for ll in range(ne):
            sx[ll] = xsrc[ll]
            sy[ll] = ysrc[ll]
        for i in numba.prange(n):
            if leaf[i] and (ns[i] > 0):
                bi = botind[i]
                ti = topind[i]
                ni = ti-bi
                tx = x[bi:ti] - xmid[i]
                ty = y[bi:ti] - ymid[i]
                loci = np.empty(ne, dtype=np.complex128)
                for ll in range(ne):
                    loci[ll] = local_expansions[i, ll]
                soli = np.zeros(ni, dtype=np.complex128)
                for ks in range(ne):
                    for kt in range(ni):
                        soli[kt] += Kernel_Eval(sx[ks], sy[ks], tx[kt], ty[kt])*loci[ks]
                for kt in range(ni):
                    sol[kt+bi] += soli[kt]

    @numba.njit(parallel=True, fastmath=True)
    def numba_target_local_expansion_evaluation(xs, ys, inds, locs, large_xs, large_ys, tree.xmids, tree.ymids, Local_Expansions):
        n = x.size
        for i in numba.prange(n):
            x = xs[i]
            y = ys[i]
            ind = inds[i]
            loc = locs[i]
            tx = x

            if leaf[i] and (ns[i] > 0):
                bi = botind[i]
                ti = topind[i]
                ni = ti-bi
                tx = x[bi:ti] - xmid[i]
                ty = y[bi:ti] - ymid[i]
                loci = np.empty(ne, dtype=np.complex128)
                for ll in range(ne):
                    loci[ll] = local_expansions[i, ll]
                soli = np.zeros(ni, dtype=np.complex128)
                for ks in range(ne):
                    for kt in range(ni):
                        soli[kt] += Kernel_Eval(sx[ks], sy[ks], tx[kt], ty[kt])*loci[ks]
                for kt in range(ni):
                    sol[kt+bi] += soli[kt]

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




    return evaluate_neighbor_interactions, numba_upwards_pass, numba_downwards_pass2

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
        out = np.empty((nt, ns), dtype=complex)
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
        out = np.empty(nt, complex)
    if isself:
        KAS(sx, sy, tau, out)
    else:
        KA(sx, sy, tx, ty, tau, out)
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
            Kern = Kernel_Form(KF, collected_equiv_xs, collected_equiv_ys, \
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
                        M2Lhere[indx,indy] = Kernel_Form(KF, small_xhere, \
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
        (evaluate_neighbor_interactions, numba_upwards_pass, numba_downwards_pass2) \
            = self.numba_functions
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
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = M2MC[ind].dot(ancestor_level.RSEQD.T).T
                numba_distribute(u_check_surfaces, temp1, ancestor_level.short_parent_ind, int(ancestor_level.n_node/4))
            numba_upwards_pass(tree.x, tree.y, Level.bot_ind, Level.top_ind, Level.ns, Level.compute_upwards, large_xs[ind], large_ys[ind], Level.xmid, Level.ymid, tau_ordered, u_check_surfaces)
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
            M2Ms = np.empty([3,3,doit.sum(),4*Nequiv], dtype=self.dtype)
            CM2Lh = CM2LS[ind+1]
            for kkx in range(3):
                for kky in range(3):
                    if not (kkx-1 == 0 and kky-1 == 0):
                        M2Ms[kkx, kky, :, :] = CM2Lh[kkx, kky].dot(descendant_level.RSEQD.T).T
            ci4 = (Level.children_ind/4).astype(int)
            numba_add_interactions(doit, ci4, Level.colleagues, Level.xmid, Level.ymid, descendant_level.Local_Solutions, M2Ms, Nequiv)
        et = time.time()
        self.print('....Time for downwards pass 1:  {:0.2f}'.format(1000*(et-st)))
        # downwards pass 2a - start at top and compute local expansions
        st = time.time()
        for ind in range(1,tree.levels):
            Level = tree.Levels[ind]
            Level.Local_Expansions = sp.linalg.lu_solve(E2C_LUs[ind], Level.Local_Solutions.T).T
        et = time.time()
        self.print('....Time for downwards pass 2a: {:0.2f}'.format(1000*(et-st)))
    def evaluate_to_sources(self):
        tree, E2C_LUs, M2MC, M2LS, CM2LS, small_xs, small_ys, large_xs, large_ys, small_radii, large_radii, widths = self._get_names()
        (evaluate_neighbor_interactions, numba_upwards_pass, numba_downwards_pass2) \
            = self.numba_functions
        if not hasattr(self, 'tau'):
            raise Exception('Need to call build_expansions first')
        if not hasattr(self, 'desorter'):
            self.desorter = np.argsort(tree.ordv)
        tau, tau_ordered = self.tau, self.tau_ordered
        solution_ordered = np.zeros_like(tau)
        # downwards pass 2b - start at top and evaluate local expansions
        st = time.time()
        for ind in range(1,tree.levels):
            Level = tree.Levels[ind]
            numba_downwards_pass2(tree.x, tree.y, Level.bot_ind, Level.top_ind, Level.ns, Level.leaf, large_xs[ind], large_ys[ind], Level.xmid, Level.ymid, Level.Local_Expansions, solution_ordered)
        et = time.time()
        self.print('....Time for downwards pass 2b: {:0.2f}'.format(1000*(et-st)))
        # downwards pass 3 - start at top and evaluate neighbor interactions
        st = time.time()
        for ind in range(1,tree.levels):
            Level = tree.Levels[ind]
            evaluate_neighbor_interactions(tree.x, tree.y, Level.leaf, Level.bot_ind, Level.top_ind, tau_ordered, Level.colleagues, solution_ordered)
        et = time.time()
        self.print('....Time for downwards pass 3:  {:0.2f}'.format(1000*(et-st)))
        return solution_ordered[self.desorter]
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
    def evaluate_to_points(self, x,  y):
        tree, E2C_LUs, M2MC, M2LS, CM2LS, small_xs, small_ys, large_xs, large_ys, small_radii, large_radii, widths = self._get_names()
        KF, KA, KAS = self.kernel_functions
        tau_ordered = self.tau_ordered
        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate local expansions
        pot = np.zeros(x.size, dtype=self.dtype)
        Local_Expansions = [Level.Local_Expansions for Level in tree.Levels]
        numba_target_local_expansion_evaluation(x, y, inds, locs, large_xs, large_ys, tree.xmids, tree.ymids, Local_Expansions)
        return inds, locs
    def _get_names(self):
        return self.tree, self.E2C_LUs, self.M2MC, self.M2LS, self.CM2LS, self.small_xs, self.small_ys, \
            self.large_xs, self.large_ys, self.small_radii, self.large_radii, self.widths



