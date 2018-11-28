import numpy as np
from .leaf import Leaf

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
        if level_restrict:
            # there's a bug in the level restriction code right now
            # it seems to be remedied by running this over and over again
            self.level_restrict()
            self.level_restrict()
            self.level_restrict()
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
        I think this routine needs some work to be fast, esp. for large trees
        """
        for LevelArray in self.LevelArrays:
            NL = len(LevelArray)
            xls = np.empty(NL)
            xhs = np.empty(NL)
            yls = np.empty(NL)
            yhs = np.empty(NL)
            for ind, node in enumerate(LevelArray):
                xls[ind] = node.xlow
                xhs[ind] = node.xhigh
                yls[ind] = node.ylow
                yhs[ind] = node.yhigh
            l1 = xls <= xhs[:,None]
            l2 = xhs >= xls[:,None]
            u1 = yls <= yhs[:,None]
            u2 = yhs >= yls[:,None]
            ll = np.logical_and(l1, l2)
            uu = np.logical_and(u1, u2)
            colleagues = np.logical_and(ll, uu)
            for ind, node in enumerate(LevelArray):
                AA = LevelArray[colleagues[:,ind]]
                node.set_colleagues(AA)
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
    def level_restrict(self):
        """
        Perform level-restriction (so that all nodes only abut other nodes
            that are at most one level finer or coarser)

        This function needs work! (both for efficiency and correctness!)
        """
        self.tag_colleagues()
        # mark all primary violators
        primary_violaters = []
        for ind1, LevelArray1 in enumerate(self.LevelArrays):
            big_violater = np.zeros(LevelArray1.shape[0], dtype=bool)
            # reduce to leaves
            leaves1 = np.array([node.leaf for node in LevelArray1])
            nleaves1 = np.sum(leaves1)
            xls1 = np.empty(nleaves1)
            xhs1 = np.empty(nleaves1)
            yls1 = np.empty(nleaves1)
            yhs1 = np.empty(nleaves1)
            violater = np.zeros(nleaves1, dtype=bool)
            for ind, node in enumerate(LevelArray1[leaves1]):
                xls1[ind] = node.xlow
                xhs1[ind] = node.xhigh
                yls1[ind] = node.ylow
                yhs1[ind] = node.yhigh
            for ind2, LevelArray2 in enumerate(self.LevelArrays):
                if ind2 > ind1 + 1:
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
            big_violater[leaves1] = violater
            primary_violaters.append(big_violater)
        # mark all secondary violators
        secondary_violaters = []
        for ind1, LevelArray1 in enumerate(self.LevelArrays):
            big_violater = np.zeros(LevelArray1.shape[0], dtype=bool)
            # reduce to leaves
            leaves1 = np.array([node.leaf for node in LevelArray1])
            nleaves1 = np.sum(leaves1)
            xls1 = np.empty(nleaves1)
            xhs1 = np.empty(nleaves1)
            yls1 = np.empty(nleaves1)
            yhs1 = np.empty(nleaves1)
            violater = np.zeros(nleaves1, dtype=bool)
            for ind, node in enumerate(LevelArray1[leaves1]):
                xls1[ind] = node.xlow
                xhs1[ind] = node.xhigh
                yls1[ind] = node.ylow
                yhs1[ind] = node.yhigh
            for ind2, LevelArray2 in enumerate(self.LevelArrays):
                if ind2 > ind1:
                    primary_violaters_here = primary_violaters[ind2]
                    np_violaters = np.sum(primary_violaters_here)
                    xls2 = np.empty(np_violaters)
                    xhs2 = np.empty(np_violaters)
                    yls2 = np.empty(np_violaters)
                    yhs2 = np.empty(np_violaters)
                    for ind, node in enumerate(LevelArray2[primary_violaters_here]):
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
                    self.tag_colleagues() #this should probably be just an update?
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

