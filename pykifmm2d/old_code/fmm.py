import numpy as np
import scipy as sp
import scipy.linalg
import time
from .leaf2 import Leaf
from .tree2 import Tree

def get_level_information(node_width, theta):
    # get information for this level
    dd = 0.1
    r1 = 0.5*node_width*(np.sqrt(2)+dd)
    r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
    small_surface_x_base = r1*np.cos(theta)
    small_surface_y_base = r1*np.sin(theta)
    large_surface_x_base = r2*np.cos(theta)
    large_surface_y_base = r2*np.sin(theta)
    return small_surface_x_base, small_surface_y_base, large_surface_x_base, \
                large_surface_y_base, r1, r2
def classify(node1, node2):
    # for two nodes at the same depth, determine relative position to
    # figure out which of the M2Ls to use
    xdist = int(round((node2.xlow - node1.xlow)/node1.xran))
    ydist = int(round((node2.ylow - node1.ylow)/node1.yran))
    closex = xdist in [-1,0,1]
    closey = ydist in [-1,0,1]
    ilist = not (closex and closey)
    return ilist, xdist, ydist

def generate_kernel_apply(kernel_form):
    def kernel_apply(sx, sy, tau, tx=None, ty=None):
        G = Kernel_Form(sx, sy, tx, ty)
        return G.dot(tau)
    return kernel_apply

def fake_print(*args, **kwargs):
    pass
def get_print_function(verbose):
    return print if verbose else fake_print

def on_the_fly_fmm(x, y, tau, Nequiv, Ncutoff, Kernel_Form, Kernel_Apply=None, \
                                                                verbose=False):
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

    if Kernel_Apply is None:
        Kernel_Apply = generate_kernel_apply(Kernel_Form)

    # build the tree
    st = time.time()
    tree = Tree(x, y, Ncutoff, level_restrict=True)
    tree_formation_time = (time.time() - st)*1000
    my_print('....Tree formed in:            {:0.1f}'.format(tree_formation_time))

    if len(tree.LevelArrays) <= 2:
        # just do a direct evaluation in this case
        solution = Kernel_Apply(x, y, tau)
    else:
        solution = _on_the_fly_fmm(tree, tau, Nequiv, Kernel_Form, \
                                                        Kernel_Apply, verbose)
    fmm_time = (time.time()-st)*1000
    my_print('FMM completed in               {:0.1f}'.format(fmm_time))
    return solution

def _on_the_fly_fmm(tree, tau, Nequiv, Kernel_Form, Kernel_Apply, verbose):
    my_print = get_print_function(verbose)

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
        width = tree.LevelArrays[ind][0].xran
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
    et = time.time()
    my_print('....Time for prep work:        {:0.2f}'.format(1000*(et-st)))
    # upwards pass - start at bottom leaf nodes and build multipoles up
    st = time.time()
    for ind in reversed(range(tree.levels)):
        all_level_nodes = np.concatenate([tree.LevelArrays[ind], tree.FakeLevelArrays[ind]])
        for node in all_level_nodes:
            if node.leaf and not node.Xlist:
                # compute S2M translation
                if node.N > 0:
                    node_xmid = node.xlow + 0.5*widths[ind]
                    node_ymid = node.ylow + 0.5*widths[ind]
                    # get check and equivalent surface
                    check_surface_x = large_xs[ind] + node_xmid
                    check_surface_y = large_ys[ind] + node_ymid
                    # get local tau values
                    tlocal = tau_ordered[node.low_ind:node.high_ind]
                    # compute solution on check surface
                    u_check_surface = Kernel_Apply(node.xhere, 
                        node.yhere, tlocal, check_surface_x, check_surface_y)
                    # get equivalent density
                    equiv_tau = sp.linalg.lu_solve(E2C_LUs[ind], u_check_surface)
                else:
                    equiv_tau = np.zeros(Nequiv)
            else:
                children = node.fake_children if node.Xlist else node.children
                # compute M2M translation from children (this will probably need to be accelerated by forming operator and doing all at once)
                eqtau0 = children[0].equiv_tau
                eqtau1 = children[1].equiv_tau
                eqtau2 = children[2].equiv_tau
                eqtau3 = children[3].equiv_tau
                eqtau = np.concatenate([eqtau0, eqtau1, eqtau2, eqtau3])
                u_check_surface = M2MC[ind].dot(eqtau)
                equiv_tau = sp.linalg.lu_solve(E2C_LUs[ind], u_check_surface)
            node.equiv_tau = equiv_tau.copy()
    et = time.time()
    my_print('....Time for upwards pass:     {:0.2f}'.format(1000*(et-st)))
    # downwards pass 1 - start at top and work down to build up local expansions
    st = time.time()
    node = tree.LevelArrays[0][0]
    node.local_solution = np.zeros(Nequiv)
    node.children[0].local_solution = np.zeros(Nequiv)
    node.children[1].local_solution = np.zeros(Nequiv)
    node.children[2].local_solution = np.zeros(Nequiv)
    node.children[3].local_solution = np.zeros(Nequiv)
    for ind in range(1,tree.levels):
        for node in tree.LevelArrays[ind]:
            # add standard interaction list multipoles to local expansion
            for parent_colleague in node.parent.colleagues:
                ilist = parent_colleague.fake_children if parent_colleague.leaf \
                            else parent_colleague.children
                for inode in ilist:
                    isilist, idx, idy = classify(node, inode)
                    if isilist:
                        node.local_solution += M2LS[ind][idx+3,idy+3].dot(inode.equiv_tau)
            if not node.leaf:
                # shift local expansion down to children
                # note the L2L matrix is the transpose of the M2M
                local_expansion = sp.linalg.lu_solve(E2C_LUs[ind], node.local_solution)
                local_solutions = M2MC[ind].T.dot(local_expansion)
                node.children[0].local_solution = local_solutions[0*Nequiv:1*Nequiv]
                node.children[1].local_solution = local_solutions[1*Nequiv:2*Nequiv]
                node.children[2].local_solution = local_solutions[2*Nequiv:3*Nequiv]
                node.children[3].local_solution = local_solutions[3*Nequiv:4*Nequiv]
    et = time.time()
    my_print('....Time for downwards pass 1: {:0.2f}'.format(1000*(et-st)))
    # downwards pass 2 - start at top and evaluate local expansions
    st = time.time()
    for ind in range(1,tree.levels):
        for node in tree.LevelArrays[ind]:
            if node.leaf:
                solution_here = solution_ordered[node.low_ind:node.high_ind]
                solution_here *= 0
                # evaluate local expansion at targets
                local_expansion = sp.linalg.lu_solve(E2C_LUs[ind], node.local_solution)
                node_xmid = node.xlow + 0.5*node.xran
                node_ymid = node.ylow + 0.5*node.yran
                equivalent_surface_x = large_xs[ind] + node_xmid
                equivalent_surface_y = large_ys[ind] + node_ymid
                solution_here += \
                    Kernel_Apply(equivalent_surface_x, equivalent_surface_y,
                        local_expansion, node.xhere, node.yhere)
    et = time.time()
    my_print('....Time for downwards pass 2: {:0.2f}'.format(1000*(et-st)))
    # downwards pass 3 - start at top and evaluate neighbor interactions
    st = time.time()
    for ind in range(tree.levels):
        for node in tree.LevelArrays[ind]:
            if node.leaf:
                solution_here = solution_ordered[node.low_ind:node.high_ind]
                # add contribution from parents colleagues children
                # this should eventually be quickened up:
                # (1) coarse neighbors don't need to be broken into fake children
                # (2) the parent itself can just be evaluated onto itself
                #   ... have to be careful not to overcount, there!
                #   ... actually this might be false because of S-List
                for parent_colleague in node.parent.colleagues:
                    ilist = parent_colleague.fake_children if parent_colleague.leaf \
                                else parent_colleague.children
                    for inode in ilist:
                        isilist, idx, idy = classify(node, inode)
                        if not isilist:
                            tau_here = tau_ordered[inode.low_ind:inode.high_ind]
                            if node is inode:
                                addition = Kernel_Apply(node.xhere, node.yhere, tau_here)
                            else:
                                addition = Kernel_Apply(inode.xhere, inode.yhere, \
                                                tau_here, node.xhere, node.yhere)
                            solution_here += addition
    et = time.time()
    my_print('....Time for downwards pass 3: {:0.2f}'.format(1000*(et-st)))
    # deorder the solution
    desorter = np.argsort(tree.ordv)
    return solution_ordered[desorter], tree



