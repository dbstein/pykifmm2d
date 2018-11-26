import numpy as np
import scipy as sp
import scipy.spatial
from .leaf2 import Leaf

"""
Hopefully updated faster version of tree
"""

class Tree(object):
    def __init__(self, x, y, ppl, level_restrict=True):
        """
        x:   array of points x
        y:   array of points y
        ppl: maximum points per leaf
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
        self.xran = self.xmax-self.xmin
        self.yran = self.ymax-self.ymin
        self.N = self.x.shape[0]
        # vector to allow reordering of density tau
        self.ordv = np.arange(self.N)
        self.LevelArrays = []
        # setup the first level
        self.LevelArrays.append(np.array([Leaf(None, self.x, self.y, \
                                self.ordv, 0, self.N, mmin, mmax, mmin, mmax)]))
        self.levels = 0
        # loop over levels and refine, if necessary
        # keep going until no leaves had to be refined
        doit = True
        while doit:
            self.levels += 1
            doit = False
            ThisLevelArray = self.LevelArrays[-1]
            NextLevelArray = []
            for node in ThisLevelArray:
                if node.N > ppl:
                    doit = True
                    new_nodes = node.get_nodes()
                    for new_node in new_nodes:
                        NextLevelArray.append(new_node)
            if doit:
                self.LevelArrays.append(np.array(NextLevelArray))
        self.collect_summary_information()
        self.tag_colleagues()
        if level_restrict:
            self.level_restrict()
        # tag all X-list objects
        self.tag_Xlist()
        # level arrays for 'fake leaves' (to avoid X-list interactions)
        self.FakeLevelArrays = []
        self.FakeLevelArrays.append([])
        # split these into the Fake Level Arrays
        self.split_Xlist()
        # gather together lots of information that will be helpful
        # for efficiently evaluating the FMM on this tree
        #### STILL TO DO!
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
        for LevelArray in self.LevelArrays:
            leaves = np.array([node.leaf for node in LevelArray])
            nleaves = np.sum(leaves)
            xls = np.empty(nleaves)
            xhs = np.empty(nleaves)
            yls = np.empty(nleaves)
            yhs = np.empty(nleaves)
            for ind, node in enumerate(LevelArray[leaves]):
                xls[ind] = node.xlow
                xhs[ind] = node.xhigh
                yls[ind] = node.ylow
                yhs[ind] = node.yhigh
            lines = []
            lines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
            lc = mpl.collections.LineCollection(lines, colors='black')
            ax.add_collection(lc)
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
    def print_structure(self):
        """
        Prints the stucture of the array (# levels, # leaves per level)
        """
        for ind, LevelArray in enumerate(self.LevelArrays):
            print('Level', ind+1, 'of', self.levels, 'has', len(LevelArray), 'nodes.')
    def tag_colleagues(self):
        """
        Tag colleagues (neighbors at same level) for every node in tree
        """
        for ind1, LevelArray in enumerate(self.LevelArrays):
            midpoint_tree = self.MidPointKDTrees[ind1]
            dist = 1.5*self.widths[ind1]
            colleague_list = midpoint_tree.query_ball_tree(midpoint_tree, dist)
            for ind2, node in enumerate(LevelArray):
                node.set_colleagues(LevelArray[colleague_list[ind2]])
    def retag_colleagues(self, lev):
        """
        Tag colleagues (neighbors at same level) for every node at a level
        """
        LevelArray = self.LevelArrays[lev]
        midpoint_tree = self.MidPointKDTrees[lev]
        dist = 1.5*self.widths[lev]
        colleague_list = midpoint_tree.query_ball_tree(midpoint_tree, dist)
        for ind, node in enumerate(LevelArray):
            node.set_colleagues(LevelArray[colleague_list[ind]])
    def collect_summary_information(self):
        # get number of nodes, width at each level
        self.Level_Ns = np.empty(self.levels, dtype=int)
        self.widths   = np.empty(self.levels, dtype=float)
        for ind, LevelArray in enumerate(self.LevelArrays):
            self.Level_Ns[ind] = len(LevelArray)
            self.widths[ind]   = LevelArray[0].xran
        # get leaf indicator variable
        self.Is_Leaf = np.empty(self.levels, dtype=object)
        self.N_Leaves = np.empty(self.levels, dtype=int)
        self.X_Lows  = np.empty(self.levels, dtype=object)
        self.X_Mids  = np.empty(self.levels, dtype=object)
        self.X_Highs = np.empty(self.levels, dtype=object)
        self.Y_Lows  = np.empty(self.levels, dtype=object)
        self.Y_Mids  = np.empty(self.levels, dtype=object)
        self.Y_Highs = np.empty(self.levels, dtype=object)
        for ind1, LevelArray in enumerate(self.LevelArrays):
            leaf_indicator = np.empty(self.Level_Ns[ind1], dtype=bool)
            x_low = np.empty(self.Level_Ns[ind1], dtype=float)
            y_low = np.empty(self.Level_Ns[ind1], dtype=float)
            for ind2, node in enumerate(LevelArray):
                leaf_indicator[ind2] = node.leaf
                x_low[ind2]  = node.xlow
                y_low[ind2]  = node.ylow
            self.Is_Leaf[ind1] = leaf_indicator
            self.N_Leaves[ind1] = np.sum(leaf_indicator)
            self.X_Lows[ind1]  = x_low
            self.X_Mids[ind1]  = x_low + 0.5*self.widths[ind1]
            self.X_Highs[ind1] = x_low + self.widths[ind1]
            self.Y_Lows[ind1]  = y_low
            self.Y_Mids[ind1]  = y_low + 0.5*self.widths[ind1]
            self.Y_Highs[ind1] = y_low + self.widths[ind1]
        # get KDTree of the midpoints
        self.MidPointKDTrees = np.empty(self.levels, dtype=object)
        for ind in range(self.levels):
            midpoint_data = np.column_stack([self.X_Lows[ind], self.Y_Lows[ind]])
            midpoint_tree = sp.spatial.cKDTree(midpoint_data)
            self.MidPointKDTrees[ind] = midpoint_tree
    def recollect_summary_information(self, lev):
        LevelArray = self.LevelArrays[lev]
        self.Level_Ns[lev] = len(LevelArray)
        leaf_indicator = np.empty(self.Level_Ns[lev], dtype=bool)
        x_low = np.empty(self.Level_Ns[lev], dtype=float)
        y_low = np.empty(self.Level_Ns[lev], dtype=float)
        for ind, node in enumerate(LevelArray):
            leaf_indicator[ind] = node.leaf
            x_low[ind] = node.xlow
            y_low[ind] = node.ylow
        self.Is_Leaf[lev] = leaf_indicator
        self.N_Leaves[lev] = np.sum(leaf_indicator)
        self.X_Lows[lev]  = x_low
        self.X_Mids[lev]  = x_low + 0.5*self.widths[lev]
        self.X_Highs[lev] = x_low + self.widths[lev]
        self.Y_Lows[lev]  = y_low
        self.Y_Mids[lev]  = y_low + 0.5*self.widths[lev]
        self.Y_Highs[lev] = y_low + self.widths[lev]
        midpoint_data = np.column_stack([self.X_Lows[lev], self.Y_Lows[lev]])
        midpoint_tree = sp.spatial.cKDTree(midpoint_data)
        self.MidPointKDTrees[lev] = midpoint_tree
    def level_restrict_old(self):
        """
        Perform level-restriction (so that all nodes only abut other nodes
            that are at most one level finer or coarser)

        This function needs work for efficiency
        """
        self.tag_colleagues()
        # mark all primary violators
        primary_violaters = []
        for ind1, LevelArray1 in enumerate(self.LevelArrays):
            big_violater = np.zeros(LevelArray1.shape[0], dtype=bool)
            # reduce to leaves
            leaves1 = self.Is_Leaf[ind1]
            nleaves1 = self.N_Leaves[ind1]
            xls1 = self.X_Lows[ind1][leaves1]
            xhs1 = self.X_Highs[ind1][leaves1]
            yls1 = self.Y_Lows[ind1][leaves1]
            yhs1 = self.Y_Highs[ind1][leaves1]
            violater = np.zeros(nleaves1, dtype=bool)
            for ind2, LevelArray2 in enumerate(self.LevelArrays):
                if ind2 > ind1 + 1:
                    leaves2 = self.Is_Leaf[ind2]
                    nleaves2 = self.N_Leaves[ind2]
                    xls2 = self.X_Lows[ind2][leaves2]
                    xhs2 = self.X_Highs[ind2][leaves2]
                    yls2 = self.Y_Lows[ind2][leaves2]
                    yhs2 = self.Y_Highs[ind2][leaves2]
                    l1 = xls1 <= xhs2[:,None]
                    l2 = xhs1 >= xls2[:,None]
                    u1 = yls1 <= yhs2[:,None]
                    u2 = yhs1 >= yls2[:,None]
                    ll = np.logical_and(l1, l2)
                    uu = np.logical_and(u1, u2)
                    touching_here = np.sum(np.logical_and(ll, uu), 0) > 0
                    violater = np.logical_or(violater, touching_here)
            big_violater[leaves1] = violater
            primary_violaters.append(big_violater)
        # mark all secondary violators
        secondary_violaters = []
        for ind1, LevelArray1 in enumerate(self.LevelArrays):
            big_violater = np.zeros(LevelArray1.shape[0], dtype=bool)
            # reduce to leaves
            leaves1 = self.Is_Leaf[ind1]
            nleaves1 = self.N_Leaves[ind1]
            xls1 = self.X_Lows[ind1][leaves1]
            xhs1 = self.X_Highs[ind1][leaves1]
            yls1 = self.Y_Lows[ind1][leaves1]
            yhs1 = self.Y_Highs[ind1][leaves1]
            violater = np.zeros(nleaves1, dtype=bool)
            for ind2, LevelArray2 in enumerate(self.LevelArrays):
                if ind2 > ind1:
                    primary_violaters_here = primary_violaters[ind2]
                    np_violaters = np.sum(primary_violaters_here)
                    leaves2 = self.Is_Leaf[ind2]
                    nleaves2 = self.N_Leaves[ind2]
                    xls2 = self.X_Lows[ind2][leaves2]
                    xhs2 = self.X_Highs[ind2][leaves2]
                    yls2 = self.Y_Lows[ind2][leaves2]
                    yhs2 = self.Y_Highs[ind2][leaves2]
                    l1 = xls1 <= xhs2[:,None]
                    l2 = xhs1 >= xls2[:,None]
                    u1 = yls1 <= yhs2[:,None]
                    u2 = yhs1 >= yls2[:,None]
                    ll = np.logical_and(l1, l2)
                    uu = np.logical_and(u1, u2)
                    touching_here = np.sum(np.logical_and(ll, uu), 0) > 0
                    violater = np.logical_or(violater, touching_here)
            big_violater[leaves1] = violater
            secondary_violaters.append(big_violater)
        # weed out things marked as secondary violaters that are already primary violaters
        for ind in range(self.levels):
            secondary_violaters[ind] = np.logical_and(secondary_violaters[ind], np.logical_not(primary_violaters[ind]))
        # divide all of the secondary violaters
        NewSecondaryNodes = [[]]*self.levels
        for ind, LevelArray in enumerate(self.LevelArrays):
            if ind < self.levels-1:
                NewLevelNodes = []
                division_leaves = LevelArray[secondary_violaters[ind]]
                for node in division_leaves:
                    NewLevelNodes.extend(node.get_nodes())
                NewSecondaryNodes[ind+1] = NewLevelNodes
        # divide all of the primary violaters
        NewPrimaryNodes = [[]]*self.levels
        for ind, LevelArray in enumerate(self.LevelArrays):
            if ind < self.levels-1:
                NewLevelNodes = []
                division_leaves = LevelArray[primary_violaters[ind]]
                for node in division_leaves:
                    NewLevelNodes.extend(node.get_nodes())
                NewPrimaryNodes[ind+1] = NewLevelNodes
        # add all of these to the actual tree
        for ind in np.arange(self.levels):
            self.LevelArrays[ind] = np.append(self.LevelArrays[ind], NewSecondaryNodes[ind])
            self.LevelArrays[ind] = np.append(self.LevelArrays[ind], NewPrimaryNodes[ind])
        self.collect_summary_information()
        self.tag_colleagues()
        # now make a downward pass on the NewPrimaryNodes and split those that still need splitting
        for ind1 in np.arange(self.levels):
            NPN = NewPrimaryNodes[ind1]
            NL = len(NPN)
            if NL > 0:
                xls1 = np.empty(NL)
                xhs1 = np.empty(NL)
                yls1 = np.empty(NL)
                yhs1 = np.empty(NL)
                violater = np.zeros(NL, dtype=bool)
                for ind, node in enumerate(NPN):
                    xls1[ind] = node.xlow
                    xhs1[ind] = node.xhigh
                    yls1[ind] = node.ylow
                    yhs1[ind] = node.yhigh
                for ind2 in np.arange(ind1+2,self.levels):
                    LevelArray2 = self.LevelArrays[ind2]
                    leaves2 = np.array([node.leaf for node in LevelArray2])
                    nleaves2 = np.sum(leaves2)
                    xls2 = np.empty(nleaves2)
                    xhs2 = np.empty(nleaves2)
                    yls2 = np.empty(nleaves2)
                    yhs2 = np.empty(nleaves2)
                    for ind, node in enumerate(LevelArray2[leaves2]):
                        xls2[ind] = node.xlow
                        xhs2[ind] = node.xhigh
                        yls2[ind] = node.ylow
                        yhs2[ind] = node.yhigh
                    l1 = xls1 <= xhs2[:,None]
                    l2 = xhs1 >= xls2[:,None]
                    u1 = yls1 <= yhs2[:,None]
                    u2 = yhs1 >= yls2[:,None]
                    ll = np.logical_and(l1, l2)
                    uu = np.logical_and(u1, u2)
                    touching_here = np.sum(np.logical_and(ll, uu), 0) > 0
                    violater = np.logical_or(violater, touching_here)
                NPNS = np.array(NPN)[violater]
                NSPLIT = NPNS.shape[0]
                if NSPLIT > 0:
                    NewNewNodes = []
                    for node in NPNS:
                        NewNewNodes.extend(node.get_nodes())
                    self.LevelArrays[ind1+1] = np.append(self.LevelArrays[ind1+1], NewNewNodes)
                    NewPrimaryNodes[ind1+1].extend(NewNewNodes)
                    # can we figure out a way to NOT do this?????
                    self.collect_summary_information()
                    self.tag_colleagues()
    def level_restrict(self):
        """
        Perform level-restriction (so that all nodes only abut other nodes
            that are at most one level finer or coarser)

        This function needs work for efficiency
        """
        self.retag_colleagues(0)
        # do these steps twice, to ensure we catch secondary violoaters
        new_nodes = 1
        while(new_nodes > 0):
            print(new_nodes)
            new_nodes = 0
            # make a downward pass, splitting all violaters
            for ind in range(1, self.levels-2):
                LevelArray = self.LevelArrays[ind]
                # first mark all violaters
                bads = np.zeros(self.Level_Ns[ind], dtype=bool)
                for ind2, node in enumerate(LevelArray):
                    if node.leaf:
                        for colleague in node.colleagues:
                            if not colleague.leaf and not bads[ind2]:
                                for child in colleague.children:
                                    if not child.leaf and not bads[ind2]:
                                        bads[ind2] = check_if_touching(node, child)
                # split the violaters at this level
                New_Children = []
                for ind2, node in enumerate(LevelArray):
                    if bads[ind2]:
                        new_children = node.get_nodes()
                        New_Children.extend(new_children)
                new_nodes += len(New_Children)
                self.LevelArrays[ind+1] = np.append(self.LevelArrays[ind+1], New_Children)
                # rebuild summary information for the NEXT level
                self.recollect_summary_information(ind+1)
                self.retag_colleagues(ind+1)
    def tag_Xlist(self):
        """
        Tag X-list leaves (that is, leaves that abut a finer leaf)
        """
        for LevelArray in self.LevelArrays[1:]:
            for node in LevelArray:
                if node.leaf:
                    any_colleague_refined = False
                    for colleague in node.colleagues:
                        any_colleague_refined = \
                                    any_colleague_refined or not colleague.leaf
                    if any_colleague_refined:
                        node.Xlist = True
    def split_Xlist(self):
        """
        Split all X-list nodes, creating 'fake children' for them
        """
        for ind, LevelArray in enumerate(self.LevelArrays[:-1]):
            NewNodes = []
            for node in LevelArray:
                if node.Xlist and not hasattr(node, 'fake_children'):
                    NewNodes.extend(node.get_nodes(fake=True))
            self.FakeLevelArrays.append(NewNodes)
    def generate_interaction_lists(self):
        for ind, LevelArray in enumerate(self.LevelArrays[:-1]):
            for node in LevelArray:
                pass

def check_if_touching(node1, node2):
    l1 = node1.xlow  <= node2.xhigh
    l2 = node1.xhigh >= node2.xlow
    u1 = node1.ylow  <= node2.yhigh
    u2 = node1.yhigh >= node2.ylow
    return l1 and l2 and u1 and u2

