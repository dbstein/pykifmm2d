import numpy as np

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
    def get_node_sel(self, xcode, ycode):
        xlow = self.xlow + xcode*self.xran/2.0
        xhigh = self.xlow + (xcode+1)*self.xran/2.0
        ylow = self.ylow + ycode*self.yran/2.0
        yhigh = self.ylow + (ycode+1)*self.yran/2.0
        selx = np.logical_and(self.xhere >= xlow, self.xhere < xhigh)
        sely = np.logical_and(self.yhere >= ylow, self.yhere < yhigh)
        sel = np.logical_and(selx, sely)
        return sel, np.sum(sel), xlow, xhigh, ylow, yhigh
    def reorder(self, vec, sels, Ns):
        sel_00, sel_01, sel_10, sel_11 = sels
        N0, N1, N2, N3, N4 = Ns
        temp00 = vec[sel_00]
        temp01 = vec[sel_01]
        temp10 = vec[sel_10]
        temp11 = vec[sel_11]
        vec[N0:N1] = temp00
        vec[N1:N2] = temp01
        vec[N2:N3] = temp10
        vec[N3:N4] = temp11
    def get_nodes(self, fake=False):
        # get indeces and counts for child nodes
        sel_00, N00, xl00, xh00, yl00, yh00 = self.get_node_sel(0,0)
        sel_01, N01, xl01, xh01, yl01, yh01 = self.get_node_sel(0,1)
        sel_10, N10, xl10, xh10, yl10, yh10 = self.get_node_sel(1,0)
        sel_11, N11, xl11, xh11, yl11, yh11 = self.get_node_sel(1,1)
        sels = (sel_00, sel_01, sel_10, sel_11)
        N0 = 0
        N1 = N00
        N2 = N00 + N01
        N3 = N00 + N01 + N10
        N4 = N00 + N01 + N10 + N11
        Ns = (N0, N1, N2, N3, N4)
        self.reorder(self.xhere, sels, Ns)
        self.reorder(self.yhere, sels, Ns)
        self.reorder(self.ordvhere, sels, Ns)
        node_00 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+N0, self.low_ind+N1, xl00, xh00, yl00, yh00)
        node_01 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+N1, self.low_ind+N2, xl01, xh01, yl01, yh01)
        node_10 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+N2, self.low_ind+N3, xl10, xh10, yl10, yh10)
        node_11 = Leaf(self, self.x, self.y, self.ordv, self.low_ind+N3, self.low_ind+N4, xl11, xh11, yl11, yh11)
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
