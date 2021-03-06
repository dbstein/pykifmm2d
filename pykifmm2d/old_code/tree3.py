import numpy as np
import scipy as sp
import numba
import scipy.spatial

"""
Hopefully updated faster version of tree
"""

tree_search_crossover = 100

@numba.njit("(f8[:],f8[:],f8,f8,i8,i1[:],i8[4])")
def classify(x, y, midx, midy, n, cl, ns):
    """
    Determine which 'class' each point belongs to in the current node
    class = 1 ==> lower left  (x in [xmin, xmid], y in [ymin, ymid])
    class = 2 ==> upper left  (x in [xmin, xmid], y in [ymid, ymax])
    class = 3 ==> lower right (x in [xmid, xmax], y in [ymin, ymid])
    class = 4 ==> upper right (x in [xmid, xmax], y in [ymid, ymax])
    inputs: 
        x,    f8[n], x coordinates for node
        y,    f8[n], y coordinates for node
        midx, f8,    x midpoint of node
        midy, f8,    y midpoint of node
        n,    i8,    number of points in node
    outputs:
        cl,   i1[n], class (described above)
        ns,   i8[4], number of points in each class
    """
    for i in range(n):
        highx = x[i] > midx
        highy = y[i] > midy
        cla = 2*highx + highy
        cl[i] = cla
        ns[cla] += 1
@numba.njit("i1(i8,i8[4])")
def get_target(i, nns):
    """
    Used in the reordering routine, determines which 'class' we
    actually want in the given index [i]. See more details in the 
    description of reorder_inplace
    inputs:
        i,   i8,    index
        nns, i8[4], cumulative sum of number of points in each class
    returns:
        
    """
    if i < nns[1]:
        target = 0
    elif i < nns[2]:
        target = 1
    elif i < nns[3]:
        target = 2
    else:
        target = 3
    return target
@numba.njit(["(f8[:],i8,i8)", "(i1[:],i8,i8)", "(i8[:],i8,i8)"])
def swap(x, i, j):
    """
    Inputs:
        x, f8[:]/i1[:]/i8[:], array on which to swap inds
        i, i8,                first index to swap
        j, i8,                second index to swap
    This function just modifies x to swap x[i] and x[j]
    """
    a = x[i]
    x[i] = x[j]
    x[j] = a
@numba.njit("i8[4](i8[4])")
def get_inds(ns):
    """
    Compute cumulative sum of the number of points in each subnode
    Inputs:
        ns,   i8[4], number of points in each subnode
    Outputs:
        inds, i8[4], cumulative sum of ns, ignoring the last part of the sum
                        i.e. if ns=[1,2,3,4], inds==>[0,1,3,6]
    """
    inds = np.empty(4, dtype=np.int64)
    inds[0] = 0
    for i in range(3):
        inds[i+1] = inds[i] + ns[i]
    return inds
@numba.njit("i8[4](f8[:],f8[:],i8[:],f8,f8)")
def reorder_inplace(x, y, ordv, midx, midy):
    """
    This function plays an integral role in tree formation and does
        several things:
        1) Determine which points belong to which subnodes
        2) Reorder x, y, and ordv variables
        3) Compute the number of points in each new subnode
    Inputs:
        x,    f8[:], x coordinates in node
        y,    f8[:], y coordinates in node
        ordv, i8[:], ordering variable
        midx, f8,    midpoint (in x) of node being split
        midy, f8,    midpoint (in y) of node being split
    Outputs:
        ns,   i8[4], number of points in each node
    """
    n = x.shape[0]
    cl = np.empty(n, dtype=np.int8)
    ns = np.zeros(4, dtype=np.int64)
    classify(x, y, midx, midy, n, cl, ns)
    inds = get_inds(ns)
    nns = inds.copy()
    for i in range(n):
        target = get_target(i, nns)
        keep_going = True
        while keep_going:
            cli = cl[i]
            icli = inds[cli]
            if cli != target:
                swap(cl, i, icli)
                swap(x, i, icli)
                swap(y, i, icli)
                swap(ordv, i, icli)
                inds[cli] += 1
                if cl[icli] == target:
                    keep_going = False
            else:
                inds[target] += 1
                keep_going = False
    return nns

@numba.njit("(f8[:],f8[:],i8[:],b1[:],f8,f8[:],f8[:],i8[:],i8[:],b1[:],f8[:],f8[:],i8[:],i8[:],i8[:],i8[:],i8,b1)", parallel=True)
def divide_and_reorder(x, y, ordv, tosplit, half_width, xmid, ymid, bot_ind, \
                top_ind, leaf, new_xmin, new_ymin, new_bot_ind, new_top_ind, parent_ind, children_ind, children_start_ind, forreal):
    """
    For every node in a level, check if node has too many points
    If it does, split that node, reordering x, y, ordv variables as we go
    Keep track of information relating to new child nodes formed
    Inputs:
        x,           f8[:], x coordinates (for whole tree)
        y,           f8[:], y coordinates (for whole tree)
        ordv,        i8[:], ordering variable (for whole tree)
        tosplit,     b1[:], whether the given node needs to be split
        half_width,  f8,    half width of current nodes
        xmid,        f8[:], x midpoints of the current nodes
        ymid,        f8[:], y midpoints of the current nodes
        bot_ind,     i8[:], bottom indeces into x/y arrays for current nodes
        top_ind,     i8[:], top indeces into x/y arrays for current nodes
    Outputs:
        leaf,         b1[:], indicator for whether current nodes are leaves
        new_xmin,     f8[:], minimum x values for child nodes
        new_ymin,     f8[:], minimum y values for child nodes
        new_bot_ind,  i8[:], bottom indeces into x/y arrays for current nodes
        new_top_ind,  i8[:], top indeces into x/y arrays for current nodes
        parent_ind,   i8[:], indeces into prior level array for parents
        children_ind, i8[:], indeces into next level array for children
        children_start_ind, i8[:], base value for children_ind, used for additions
        forreal, b1: whether this division is real (False for Xlist!)
    """
    num_nodes = xmid.shape[0]
    split_ids = np.zeros(num_nodes, dtype=np.int64)
    split_tracker = 0
    for i in range(num_nodes):
        if tosplit[i]:
            split_ids[i] = split_tracker
            split_tracker += 1
    for i in numba.prange(num_nodes):
        if tosplit[i]:
            split_tracker = split_ids[i]
            bi = bot_ind[i]
            ti = top_ind[i]
            nns = reorder_inplace(x[bi:ti], y[bi:ti], ordv[bi:ti], xmid[i], ymid[i])
            new_xmin[4*split_tracker + 0] = xmid[i] - half_width
            new_xmin[4*split_tracker + 1] = xmid[i] - half_width
            new_xmin[4*split_tracker + 2] = xmid[i]
            new_xmin[4*split_tracker + 3] = xmid[i]
            new_ymin[4*split_tracker + 0] = ymid[i] - half_width
            new_ymin[4*split_tracker + 1] = ymid[i]
            new_ymin[4*split_tracker + 2] = ymid[i] - half_width
            new_ymin[4*split_tracker + 3] = ymid[i]
            new_bot_ind[4*split_tracker + 0] = bot_ind[i] + nns[0]
            new_bot_ind[4*split_tracker + 1] = bot_ind[i] + nns[1]
            new_bot_ind[4*split_tracker + 2] = bot_ind[i] + nns[2]
            new_bot_ind[4*split_tracker + 3] = bot_ind[i] + nns[3]
            new_top_ind[4*split_tracker + 0] = bot_ind[i] + nns[1]
            new_top_ind[4*split_tracker + 1] = bot_ind[i] + nns[2]
            new_top_ind[4*split_tracker + 2] = bot_ind[i] + nns[3]
            new_top_ind[4*split_tracker + 3] = top_ind[i]
            if forreal:
                leaf[i] = False
            for j in range(4):
                parent_ind[4*split_tracker + j] = i
            children_ind[i] = children_start_ind + 4*split_tracker

def get_new_level(level, x, y, ordv, ppl):
    """
    Split any nodes in level that have more than ppl points
    Into new nodes, reordering x/y/ordv along the way
    And construct new level from each node
    Inputs:
        level, Level
        x,     f8[:], x coordinates (for whole tree)
        y,     f8[:], y coordinates (for whole tree)
        ordv,  i8[:], ordering variable (for whole tree)
        ppl,   i8,    number of points per leaf that triggers refinement
    """
    # figure out how many need to be split
    to_split = level.ns > ppl
    num_to_split = to_split.sum()
    num_new = 4*num_to_split
    # allocate memory for outputs of divide_and_reorder
    xmin = np.empty(num_new, dtype=float)
    ymin = np.empty(num_new, dtype=float)
    bot_ind = np.empty(num_new, dtype=int)
    top_ind = np.empty(num_new, dtype=int)
    parent_ind = np.empty(num_new, dtype=int)
    # divde current nodes and reorder the x, y, and ordv arrays
    divide_and_reorder(x, y, ordv, to_split, level.half_width, level.xmid, level.ymid, \
            level.bot_ind, level.top_ind, level.leaf, xmin, ymin, bot_ind, top_ind, parent_ind, level.children_ind, 0, True)
    # construct new level
    new_level = Level(xmin, ymin, level.half_width, bot_ind, top_ind, parent_ind)
    # determine whether further refinement is needed
    keep_going = np.any(new_level.ns > ppl)
    return new_level, keep_going

@numba.njit("(f8[:],f8[:],i8[:,:],f8)", parallel=True)
def numba_tag_colleagues(xmid, ymid, colleagues, dist):
    n = xmid.shape[0]
    dist2 = dist*dist
    for i in numba.prange(n):
        itrack = 0
        for j in range(n):
            dx = xmid[i]-xmid[j]
            dy = ymid[i]-ymid[j]
            d2 = dx*dx + dy*dy
            if d2 < dist2:
                colleagues[i,itrack] = j
                itrack += 1

@numba.njit("f8[:],f8[:],f8,i8[:],i8[:,:],b1[:],i8[:],i8[:,:]", parallel=True)
def numba_loop_colleagues(xmid, ymid, dist, parent_ind, ancestor_colleagues, 
                                ancestor_leaf, ancestor_child_inds, colleagues):
    n = xmid.shape[0]
    dist2 = dist*dist
    for i in numba.prange(n):
        itrack = 0
        pi = parent_ind[i]
        for j in range(9):
            pij = ancestor_colleagues[pi,j]
            if pij >= 0:
                if not ancestor_leaf[pij]:
                    ck = ancestor_child_inds[pij]
                    for k in range(4):
                        ckk = ck + k                  
                        dx = xmid[i]-xmid[ckk]
                        dy = ymid[i]-ymid[ckk]
                        d2 = dx*dx + dy*dy
                        if d2 < dist2:
                            colleagues[i,itrack] = ckk
                            itrack += 1

def split_bad_leaves(Level, Descendant_Level, x, y, ordv, bads):
    num_to_split = bads.sum()
    num_new = 4*num_to_split
    # allocate memory for outputs of divide_and_reorder
    xmin = np.empty(num_new, dtype=float)
    ymin = np.empty(num_new, dtype=float)
    bot_ind = np.empty(num_new, dtype=int)
    top_ind = np.empty(num_new, dtype=int)
    parent_ind = np.empty(num_new, dtype=int)
    # divde current nodes and reorder the x, y, and ordv arrays
    divide_and_reorder(x, y, ordv, bads, Level.half_width, Level.xmid, Level.ymid, \
            Level.bot_ind, Level.top_ind, Level.leaf, xmin, ymin, bot_ind, top_ind, parent_ind, Level.children_ind, Descendant_Level.n_node, True)
    # add these new nodes to the descendent level
    Descendant_Level.add_nodes(xmin, ymin, bot_ind, top_ind, parent_ind)
    # retag colleagues of the descendant level
    Descendant_Level.tag_colleagues(Level)

def split_Xlist(Level, Descendant_Level, x, y, ordv, Xlist):
    num_to_split = Xlist.sum()
    num_new = 4*num_to_split
    # allocate memory for outputs of divide_and_reorder
    xmin = np.empty(num_new, dtype=float)
    ymin = np.empty(num_new, dtype=float)
    bot_ind = np.empty(num_new, dtype=int)
    top_ind = np.empty(num_new, dtype=int)
    parent_ind = np.empty(num_new, dtype=int)
    # allocate fake children array pointer
    Level.fake_children_ind = -np.ones(Level.n_node, dtype=int)
    # divde current nodes and reorder the x, y, and ordv arrays
    divide_and_reorder(x, y, ordv, Xlist, Level.half_width, Level.xmid, Level.ymid, \
            Level.bot_ind, Level.top_ind, Level.leaf, xmin, ymin, bot_ind, top_ind, parent_ind, Level.fake_children_ind, 0, False)
    # add these new nodes to the descendent level
    Descendant_Level.add_fake_nodes(xmin, ymin, bot_ind, top_ind, parent_ind)

class Level(object):
    """
    Set of nodes all at the same level (with same width...)
    For use in constructing Tree objects
    """
    def __init__(self, xmin, ymin, width, bot_ind, top_ind, parent_ind):
        """
        Inputs:
            xmin,       f8[:], minimum x values for each node
            ymin,       f8[:], minimum y values for each node
            width,      f8,    width of each node (must be same in x/y directions)
            bot_ind,    i8[:], bottom indeces into x/y arrays for current nodes
            top_ind,    i8[:], top indeces into x/y arrays for current nodes
            parent_ind, i8[:], index to find parent in prior level array
        """
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.half_width = 0.5*self.width
        self.bot_ind = bot_ind
        self.top_ind = top_ind
        self.parent_ind = parent_ind
        self.basic_computations()
        self.leaf = np.ones(self.n_node, dtype=bool)
        self.children_ind = -np.ones(self.n_node, dtype=int)
        self.has_fake_children = False
    def basic_computations(self):
        self.ns = self.top_ind - self.bot_ind
        self.xmid = self.xmin + self.half_width
        self.ymid = self.ymin + self.half_width
        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.width
        self.n_node = self.xmin.shape[0]
        self.short_parent_ind = self.parent_ind[::4]
    def tag_colleagues(self, ancestor=None):
        self.colleagues = -np.ones([self.n_node, 9], dtype=int)
        if self.n_node < tree_search_crossover:
            self.direct_tag_colleagues()
        elif ancestor is None:
            self.ckdtree_tag_colleagues()
        else:
            self.ancestor_tag_colleagues(ancestor)
    def direct_tag_colleagues(self):
        dist = 1.5*self.width
        numba_tag_colleagues(self.xmid, self.ymid, self.colleagues, dist)
    def ckdtree_tag_colleagues(self):
        dist = 1.5*self.width
        self.construct_midpoint_tree()
        colleague_list = self.midpoint_tree.query_ball_tree(self.midpoint_tree, dist)
        for ind in range(self.n_node):
            clist = colleague_list[ind]
            self.colleagues[ind,:len(clist)] = clist
    def ancestor_tag_colleagues(self, ancestor):
        dist = 1.5*self.width
        numba_loop_colleagues(self.xmid, self.ymid, dist, self.parent_ind, 
            ancestor.colleagues, ancestor.leaf, ancestor.children_ind, self.colleagues)
    def construct_midpoint_tree(self):
        if not hasattr(self, 'midpoint_tree'):
            midpoint_data = np.column_stack([self.xmid, self.ymid])
            self.midpoint_tree = sp.spatial.cKDTree(midpoint_data)
    def get_depths(self, descendant):
        self.depths = np.zeros(self.n_node, dtype=int)
        if descendant is not None:
            numba_get_depths(self.depths, self.leaf, self.children_ind, descendant.depths)
    def add_nodes(self, xmin, ymin, bot_ind, top_ind, parent_ind):
        self.xmin = np.concatenate([self.xmin, xmin])
        self.ymin = np.concatenate([self.ymin, ymin])
        self.bot_ind = np.concatenate([self.bot_ind, bot_ind])
        self.top_ind = np.concatenate([self.top_ind, top_ind])
        self.parent_ind = np.concatenate([self.parent_ind, parent_ind])
        self.leaf = np.concatenate([self.leaf, np.ones(xmin.shape[0], dtype=bool)])
        self.children_ind = np.concatenate([self.children_ind, -np.ones(xmin.shape[0], dtype=int)])
        self.basic_computations()
    def add_fake_nodes(self, xmin, ymin, bot_ind, top_ind, parent_ind):
        self.fake_xmin = xmin
        self.fake_ymin = ymin
        self.fake_bot_ind = bot_ind
        self.fake_top_ind = top_ind
        self.fake_parent_ind = parent_ind
        self.fake_leaf = np.ones(xmin.shape[0], dtype=bool)
        self.fake_ns = self.fake_top_ind - self.fake_bot_ind
        self.fake_xmid = self.fake_xmin + self.half_width
        self.fake_ymid = self.fake_ymin + self.half_width
        self.fake_xmax = self.fake_xmin + self.width
        self.fake_ymax = self.fake_ymin + self.width
        self.n_fake_node = self.fake_xmin.shape[0]
        self.has_fake_children = True
        self.fake_short_parent_ind = self.fake_parent_ind[::4]
    def get_Xlist(self):
        self.Xlist = np.zeros(self.n_node, dtype=bool)
        numba_get_Xlist(self.depths, self.colleagues, self.leaf, self.Xlist)
    def add_null_Xlist(self):
        self.Xlist = np.zeros(self.n_node, dtype=bool)
    def allocate_workspace(self, Nequiv):
        self.Local_Expansions = np.zeros([self.n_node, Nequiv], dtype=float)
        self.Local_Solutions = np.zeros([self.n_node, Nequiv], dtype=float)
        self.Check_Us = np.zeros([self.n_node, Nequiv], dtype=float)
        self.Equiv_Densities = np.zeros([self.n_node, Nequiv], dtype=float)
        resh = (int(self.n_node/4), int(Nequiv*4))
        self.RSEQD = np.reshape(self.Equiv_Densities, resh)
        if self.RSEQD.flags.owndata:
            raise Exception('Something went wrong with reshaping the equivalent densities, it made a copy instead of a view.')
        if self.has_fake_children:
            self.Fake_Check_Us = np.zeros([self.n_fake_node, Nequiv], dtype=float)
            self.Fake_Equiv_Densities = np.zeros([self.n_fake_node, Nequiv], dtype=float)
            resh = (int(self.n_fake_node/4), int(Nequiv*4))
            self.Fake_RSEQD = np.reshape(self.Fake_Equiv_Densities, resh)
            if self.Fake_RSEQD.flags.owndata:
                raise Exception('Something went wrong with reshaping the equivalent densities, it made a copy instead of a view.')

@numba.njit("(i8[:],b1[:],i8[:],i8[:])",parallel=True)
def numba_get_depths(depths, leaves, children_ind, descendant_depths):
    n = depths.shape[0]
    for i in numba.prange(n):
        if not leaves[i]:
            child_depths = descendant_depths[children_ind[i]:children_ind[i]+4]
            max_child_depth = np.max(child_depths)
            depths[i] = max_child_depth + 1

@numba.njit("(i8[:],i8[:,:],b1[:],b1[:])",parallel=True)
def numba_get_bads(depths, colleagues, leaf, bads):
    n = depths.shape[0]
    for i in numba.prange(n):
        if leaf[i]:
            badi = False
            for j in range(9):
                cj = colleagues[i,j]
                if cj >= 0:
                    level_dist = depths[i]-depths[cj]
                    if level_dist > 1 or level_dist < -1:
                        badi = True
            bads[i] = badi

@numba.njit("(i8[:],i8[:,:],b1[:],b1[:])",parallel=True)
def numba_get_Xlist(depths, colleagues, leaf, Xlist):
    n = depths.shape[0]
    for i in numba.prange(n):
        if leaf[i]:
            XlistI = False
            for j in range(9):
                cj = colleagues[i,j]
                if cj >=0:
                    level_dist = depths[i]-depths[cj]
                    if level_dist < 0:
                        XlistI = True
            Xlist[i] = XlistI

class Tree(object):
    """
    Quadtree object for use in computing FMMs
    """
    def __init__(self, x, y, ppl):
        """
        Inputs:
            x,   f8[:], x coordinates for which tree will be constructed
            y,   f8[:], y coordinates for which tree will be constructed
            ppl, i8,    cutoff value that triggers leaf refinement
        """
        self.x = x.copy()
        self.y = y.copy()
        self.points_per_leaf = ppl
        xmin = self.x.min()
        xmax = self.x.max()
        ymin = self.y.min()
        ymax = self.y.min()
        mmin = int(np.floor(np.min([xmin, ymin])))
        mmax = int(np.ceil (np.max([xmax, ymax])))
        self.xmin = mmin
        self.xmax = mmax
        self.ymin = mmin
        self.ymax = mmax
        self.N = self.x.shape[0]
        self.workspace_allocated = False
        # vector to allow reordering of density tau
        self.ordv = np.arange(self.N)
        self.Levels = []
        # setup the first level
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        width = self.xmax-self.xmin
        bot_ind_arr = np.array((0,))
        top_ind_arr = np.array((self.N,))
        parent_ind_arr = np.array((-1,))
        level_0 = Level(xminarr, yminarr, width, bot_ind_arr, top_ind_arr, parent_ind_arr)
        self.Levels.append(level_0)
        if self.N > self.points_per_leaf:
            current_level = level_0
            keep_going = True
            while keep_going:
                new_level, keep_going = get_new_level(current_level, \
                                self.x, self.y, self.ordv, self.points_per_leaf)
                self.Levels.append(new_level)
                current_level = new_level
        self.levels = len(self.Levels)
        # tag colleagues
        self.tag_colleagues()
        # gather depths
        self.gather_depths()
        # perform level restriction
        self.level_restrict()
        # tag the Xlist
        self.tag_Xlist()
        # 'split' the Xlist
        self.split_Xlist()
    def plot(self, ax, mpl, points=False, **kwargs):
        """
        Create a simple plot to visualize the tree
        Inputs:
            ax,     axis, required: on which to plot things
            mpl,    handle to matplotlib import
            points, bool, optional: whether to also scatter the points
        """
        if points:
            ax.scatter(self.x, self.y, color='red', **kwargs)
        lines = []
        clines = []
        for level in self.Levels:
            nleaves = np.sum(level.leaf)
            xls = level.xmin[level.leaf]
            xhs = level.xmax[level.leaf]
            yls = level.ymin[level.leaf]
            yhs = level.ymax[level.leaf]
            lines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
        lc = mpl.collections.LineCollection(lines, colors='black')
        ax.add_collection(lc)
        try:
            for ind in range(1, self.levels-1):
                level = self.Levels[ind]
                nxlist = np.sum(level.Xlist)
                xls = level.xmin[level.Xlist]
                xms = level.xmid[level.Xlist]
                xhs = level.xmax[level.Xlist]
                yls = level.ymin[level.Xlist]
                yms = level.ymid[level.Xlist]
                yhs = level.ymax[level.Xlist]
                clines.extend([[(xms[i], yls[i]), (xms[i], yhs[i])] for i in range(nxlist)])
                clines.extend([[(xls[i], yms[i]), (xhs[i], yms[i])] for i in range(nxlist)])
            clc = mpl.collections.LineCollection(clines, colors='gray', alpha=0.25)
            ax.add_collection(clc)
        except:
            pass
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
    def print_structure(self):
        """
        Prints the stucture of the array (# levels, # leaves per level)
        """
        for ind, Level in enumerate(self.Levels):
            print('Level', ind+1, 'of', self.levels, 'has', Level.n_node, 'nodes.')
    def tag_colleagues(self):
        """
        Tag colleagues (neighbors at same level) for every node in tree
        """
        for ind, Level in enumerate(self.Levels):
            ancestor = None if ind == 0 else self.Levels[ind-1]
            Level.tag_colleagues(ancestor)
    def retag_colleagues(self, lev):
        ancestor = None if lev == 0 else self.Levels[lev-1]
        self.Levels[lev].tag_colleagues(ancestor)
    def collect_summary_information(self):
        pass
    def recollect_summary_information(self, lev):
        pass
    def level_restrict(self):
        new_nodes = 1
        while(new_nodes > 0):
            new_nodes = 0
            for ind in range(1, self.levels-2):
                Level = self.Levels[ind]
                Descendant_Level = self.Levels[ind+1]
                bads = np.zeros(Level.n_node, dtype=bool)
                numba_get_bads(Level.depths, Level.colleagues, Level.leaf, bads)
                num_bads = np.sum(bads)
                split_bad_leaves(Level, Descendant_Level, self.x, self.y, self.ordv, bads)
                self.gather_depths()
                new_nodes += num_bads
    def tag_Xlist(self):
        for ind in range(1, self.levels-1):
            Level = self.Levels[ind]
            Level.get_Xlist()
        self.Levels[0].add_null_Xlist()
        self.Levels[-1].add_null_Xlist()
    def split_Xlist(self):
        for ind in range(1, self.levels-1):
            Level = self.Levels[ind]
            Xlist = Level.Xlist
            num_Xlist = np.sum(Xlist)
            if num_Xlist > 0:
                Descendant_Level = self.Levels[ind+1]
                split_Xlist(Level, Descendant_Level, self.x, self.y, self.ordv, Xlist)
    def generate_interaction_lists(self):
        pass
    def gather_depths(self):
        for ind, Level in reversed(list(enumerate(self.Levels))):
            descendant = None if ind==self.levels-1 else self.Levels[ind+1]
            Level.get_depths(descendant)
    def allocate_workspace(self, Nequiv):
        for Level in self.Levels[1:]:
            Level.allocate_workspace(Nequiv)
        self.workspace_allocated = True
    def reorder_levels(self):
        # might not really want to do this!
        # reorder levels so that parents are in same order as the children...
        ind = self.levels-2
        Level = self.Levels[ind]
        child_level = self.Levels[ind+1]
        ancestor_level = self.Levels[ind-1]
        parent_inds = child_level.parent_ind[::4]
        fake_parent_inds = child_level.fake_parent_ind[::4]
        leaf_inds = np.logical_and(Level.leaf, np.logical_not(Level.Xlist))
        # reorder this level
        reorder_list = [
            Level.xmin,
            Level.xmid,
            Level.xmax,
            Level.ymin,
            Level.ymid,
            Level.ymax,
            Level.bot_ind,
            Level.top_ind,
            Level.ns,
        ]
        for elem in reorder_list:
            reorder(Level.xmin, parent_inds, fake_parent_inds, leaf_inds)
        # adjust parent inds at the child level
        n_parents = int(round(child_level.n_node/4))
        n_fake_parents = int(round(child_level.fake_parent_ind.shape[0]/4))
        child_level.parent_ind = np.repeat(np.arange(n_parents), 4)
        child_level.fake_parent_ind = child_level.parent_ind[-1] + 1 + np.repeat(np.arange(n_fake_parents), 4)
        # adjust child pointers at the ancestor level

def reorder(x, s1, s2, s3):
    return np.concatenate([x[s1], x[s2], x[s3]])



