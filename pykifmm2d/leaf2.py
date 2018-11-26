import numpy as np
import numba

@numba.njit("(f8[:],f8[:],i8[:],f8[:],f8[:],i8[:],i8[:],i8[:],i8[:],f8,f8)")
def divide_and_reorder(px, py, ov, pxr, pyr, ovr, cl, ns, inds, xmid, ymid):
    n = px.shape[0]
    for i in range(4):
        ns[i] = 0
    for i in range(n):
        highx = px[i] > xmid
        highy = py[i] > ymid
        cla = 2*highx + highy
        cl[i] = cla
        ns[cla] += 1
    inds[0] = 0
    inds[1] = ns[0]
    inds[2] = ns[0] + ns[1]
    inds[3] = ns[0] + ns[1] + ns[2]
    for i in range(n):
        clh = cl[i]
        indh = inds[clh]
        pxr[indh] = px[i]
        pyr[indh] = py[i]
        ovr[indh] = ov[i]
        inds[clh] += 1

class Leaf(object):
    def __init__(self, parent, x, y, ordv, low_ind, high_ind, xlow, xhigh, ylow, yhigh):
        self.parent = parent
        self.x = x
        self.y = y
        self.ordv = ordv
        self.xhere = self.x[low_ind:high_ind]
        self.yhere = self.y[low_ind:high_ind]
        self.ordvhere = self.ordv[low_ind:high_ind]
        self.low_ind = low_ind
        self.high_ind = high_ind
        self.xlow = xlow
        self.xhigh = xhigh
        self.ylow = ylow
        self.yhigh = yhigh
        self.xran = self.xhigh - self.xlow
        self.yran = self.yhigh - self.ylow
        self.xmid = self.xlow + self.xran/2.0
        self.ymid = self.ylow + self.yran/2.0
        self.N = self.high_ind - self.low_ind
        self.leaf = True
        self.Xlist = False
    def get_nodes(self, fake=False):
        # temp spaces for reordered vars
        xre = np.empty_like(self.xhere)
        yre = np.empty_like(self.yhere)
        ore = np.empty_like(self.ordvhere)
        # vars needed by the numba routine
        cl = np.zeros(self.N, dtype=int)
        ns = np.zeros(4, dtype=int)
        inds = np.zeros(4, dtype=int)
        divide_and_reorder(self.xhere, self.yhere, self.ordvhere, \
            xre, yre, ore, cl, ns, inds, self.xmid, self.ymid)
        # reorder the actual array
        # note this also affects the 'here' arrays!
        self.x[self.low_ind:self.high_ind] = xre
        self.y[self.low_ind:self.high_ind] = yre
        self.ordv[self.low_ind:self.high_ind] = ore
        NS = np.concatenate([ (0,), np.cumsum(ns) ])
        node_00 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+NS[0], self.low_ind+NS[1], self.xlow, self.xmid,  self.ylow, self.ymid)
        node_01 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+NS[1], self.low_ind+NS[2], self.xlow, self.xmid,  self.ymid, self.yhigh)
        node_10 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+NS[2], self.low_ind+NS[3], self.xmid, self.xhigh, self.ylow, self.ymid)
        node_11 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+NS[3], self.low_ind+NS[4], self.xmid, self.xhigh, self.ymid, self.yhigh)
        children = [node_00, node_01, node_10, node_11]
        if fake:
            self.fake_children = children
        else:
            self.children = children
            self.leaf = False
        return children        
    def wipe_colleagues(self):
        self.colleagues = []
    def set_colleagues(self, nodes):
        self.colleagues = nodes
    def add_colleagues(self, nodes):
        self.colleagues.extend(nodes)
