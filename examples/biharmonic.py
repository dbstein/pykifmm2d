import numpy as np
from scipy.special import struve, y0, yn
from fast_interp import FunctionGenerator as FG

def protect(F):
	def new_func(x):
		if isinstance(x, np.ndarray):
			dtype = type(F(1))
			sel = x > 1e-10
			out = np.zeros(x.shape, dtype=dtype)
			out[sel] = F(x[sel])
		else:
			out = 0.0 if x < 1e-10 else F(x)
		return out
	return new_func

H0 = FG(lambda x: struve( 0, x), 1e-10, 10)
H1 = FG(lambda x: struve(-1, x), 1e-10, 10)
H2 = FG(lambda x: struve(-2, x), 1e-10, 10)
H3 = FG(lambda x: struve(-3, x), 1e-10, 10)
Y0 = FG(lambda x: y0(x),         1e-10, 10)
Y1 = FG(lambda x: yn(1, x),      1e-10, 10)
Y2 = FG(lambda x: yn(2, x),      1e-10, 10)

"""
Test how to do biharmonic FMM using cloud of points...
"""

def random3(N):
	a1 = np.random.rand(N)-0.5
	a2 = np.random.rand(N)-0.5
	a3 = np.random.rand(N)-0.5
	return a1, a2, a3

# number of points to compute for
N = 100
# number of points in check/equiv surfaces
N_equiv = 64    # for 2D tests
N_order = 16    # for 3D tests
inplane = False # if to only use in-plane sources/targs for 3D tests

# construct some data to run FMM on
sx, sy, sz = random3(N)
tx, ty, tz = random3(N)
tx += 2
tau = np.random.rand(N)
if inplane:
	sz *= 0.0
	tz *= 0.0

# generate check/equiv surfaces for 2D
node_width = 1.0
theta = np.linspace(0, 2*np.pi, N_equiv, endpoint=False)
dd = 0.1
r1 = 0.5*node_width*(np.sqrt(2)+dd)
r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
# normals
nx2 = np.cos(theta)
ny2 = np.sin(theta)
ex2, ey2 = r1*nx2, r1*ny2
cx2, cy2 = r2*nx2, r2*ny2

# generate check/equiv surfaces for 3D
zp = np.ones(N_order**2)
bq, wq = np.polynomial.legendre.leggauss(N_order)
bq1, bq2 = np.meshgrid(bq, bq, indexing='ij')
bq1 = bq1.ravel()
bq2 = bq2.ravel()
wqx, wqy = np.meshgrid(wq, wq, indexing='ij')
wq = (wqx*wqy).ravel()*r1**2
wq = np.concatenate([wq]*6)
# generate the nine faces
xq1, yq1, zq1 = bq1.copy(), bq2.copy(),  zp.copy() 
xq2, yq2, zq2 = bq1.copy(), bq2.copy(), -zp.copy() 
xq3, yq3, zq3 = bq1.copy(),  zp.copy(), bq2.copy()
xq4, yq4, zq4 = bq1.copy(), -zp.copy(), bq2.copy()
xq5, yq5, zq5 =  zp.copy(), bq1.copy(), bq2.copy()
xq6, yq6, zq6 = -zp.copy(), bq1.copy(), bq2.copy()
xq = np.concatenate([xq1, xq2, xq3, xq4, xq5, xq6])
yq = np.concatenate([yq1, yq2, yq3, yq4, yq5, yq6])
zq = np.concatenate([zq1, zq2, zq3, zq4, zq5, zq6])
# scale these to get equiv/check
dd = 0.1
r1 = (1+dd)*0.5
r2 = (3-2*dd)*0.5
ex3, ey3, ez3 = r1*xq, r1*yq, r1*zq
cx3, cy3, cz3 = r2*xq, r2*yq, r2*zq
# get the normals
v1 = np.ones(N_order**2)
v0 = np.zeros(N_order**2)
nx3 = np.concatenate([ v0,  v0, v0,  v0, v0,  v0 ])
ny3 = np.concatenate([ v0,  v0, v1, -v1, v0,  v0 ])
nz3 = np.concatenate([ v1, -v1, v0,  v0, v1, -v1 ])

def printit(title, name, err):
	extra_spaces = 15 - len(name)
	print('   ', title, name + ':', ' '*extra_spaces, '{:0.1e}'.format(err))

def get_r2(x1, y1, x2, y2):
	dx = x2[:,None] - x1
	dy = y2[:,None] - y1
	r = np.hypot(dx, dy)
	return dx, dy, r
def kk2(r, G):
	return G(r)
def kn2(dx, dy, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	return Gx*nx2[:,None] + Gy*ny2[:,None]
def nk2(dx, dy, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	return -(Gx*nx2 + Gy*ny2)
def nn2(dx, dy, r, Gp, Gpp):
	Gpr = Gp(r)
	Gppr = Gpp(r)
	Gxx = Gppr*dx*dx/r**2 + Gpr/r - Gpr*dx*dx/r**3
	Gxy = Gppr*dx*dy/r**2 - Gpr*dx*dy/r**3
	Gyy = Gppr*dy*dy/r**2 + Gpr/r - Gpr*dy*dy/r**3
	return -(nx2*(Gxx*nx2[:,None]+Gxy*ny2[:,None]) + ny2*(Gxy*nx2[:,None]+Gyy*ny2[:,None]))

# test function for 2D, 2nd Order
def test_2d_2O(G, name):
	G = protect(G)
	# source to target
	dx, dy, r = get_r2(sx,  sy,  tx,  ty)
	s2tG = kk2(r,  G)
	# source to check
	dx, dy, r = get_r2(sx,  sy,  cx2, cy2)
	s2lG = kk2(r, G)
	# equivalent to check
	dx, dy, r = get_r2(ex2, ey2, cx2, cy2)
	e2lG = kk2(r, G)
	 # equivalent to target
	dx, dy, r = get_r2(ex2, ey2, tx,  ty)
	e2tG = kk2(r,  G)
	# compute the direct eval
	truth = s2tG.dot(tau)
	# compute the FMM eval
	est = e2tG.dot(np.linalg.solve(e2lG, s2lG.dot(tau)))
	# check the error
	err = np.abs(truth-est).max()
	printit('2D, 2nd Order', name, err)
def test_2d_4O(G, Gp, Gpp, name):
	G = protect(G)
	Gp = protect(Gp)
	Gpp = protect(Gpp)
	# source to target
	dx, dy, r = get_r2(sx,  sy,  tx,  ty)
	s2tG   = kk2(r, G)
	# source to check
	dx, dy, r = get_r2(sx, sy, cx2, cy2)
	s2lGa  = kk2(r, G)
	s2lGb  = kn2(dx, dy, r, Gp)
	s2lG   = np.row_stack([s2lGa, s2lGb])
	# equivalent to check
	dx, dy, r = get_r2(ex2, ey2, cx2, cy2)
	e2lGaa = kk2(r, G)
	e2lGab = kn2(dx, dy, r, Gp)
	e2lGba = nk2(dx, dy, r, Gp)
	e2lGbb = nn2(dx, dy, r, Gp, Gpp)
	e2lGa  = np.row_stack([e2lGaa, e2lGab])
	e2lGb  = np.row_stack([e2lGba, e2lGbb])
	e2lG   = np.column_stack([e2lGa, e2lGb])
	# equivalent to target
	dx, dy, r = get_r2(ex2, ey2, tx, ty)
	e2tGa  = kk2(r, G)
	e2tGb  = nk2(dx, dy, r, Gp)
	e2tG   = np.column_stack([e2tGa, e2tGb])
	# compute the direct eval
	truth = s2tG.dot(tau)
	# compute the FMM eval
	est = e2tG.dot(np.linalg.solve(e2lG, s2lG.dot(tau)))
	# check the error
	err = np.abs(truth-est).max()
	printit('2D, 4th Order', name, err)

def get_r3(x1, y1, z1, x2, y2, z2):
	dx = x2[:,None] - x1
	dy = y2[:,None] - y1
	dz = z2[:,None] - z1
	r = np.sqrt(dx**2 + dy**2 + dz**2)
	return dx, dy, dz, r
def kk3(r, G):
	return G(r)
def kn3(dx, dy, dz, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	Gz = Gpr*dz/r
	return Gx*nx3[:,None] + Gy*ny3[:,None] + Gz*nz3[:,None]
def nk3(dx, dy, dz, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	Gz = Gpr*dz/r
	return -(Gx*nx3 + Gy*ny3 + Gz*nz3)
def nn3(dx, dy, dz, r, Gp, Gpp):
	Gpr = Gp(r)
	Gppr = Gpp(r)
	Gxx = Gppr*dx*dx/r**2 + Gpr/r - Gpr*dx*dx/r**3
	Gxy = Gppr*dx*dy/r**2 - Gpr*dx*dy/r**3
	Gxz = Gppr*dx*dz/r**2 - Gpr*dx*dz/r**3
	Gyy = Gppr*dy*dy/r**2 + Gpr/r - Gpr*dy*dy/r**3
	Gyz = Gppr*dy*dz/r**2 - Gpr*dy*dz/r**3
	Gzz = Gppr*dz*dz/r**2 + Gpr/r - Gpr*dz*dz/r**3
	return -(nx3*(Gxx*nx3[:,None]+Gxy*ny3[:,None]+Gxz*nz3[:,None]) + ny3*(Gxy*nx3[:,None]+Gyy*ny3[:,None]+Gyz*nz3[:,None]) + nz3*(Gxz*nx3[:,None]+Gyz*ny3[:,None]+Gzz*nz3[:,None]))

# test function for 3D, 2nd Order
def test_3d_2O(G, name):
	G = protect(G)
	# source to target
	dx, dy, dz, r = get_r3(sx,  sy,  sz,  tx,  ty,  tz)
	s2tG = kk3(r,  G)
	# source to check
	dx, dy, dz, r = get_r3(sx,  sy,  sz,  cx3, cy3, cz3)
	s2lG = kk3(r, G)
	# equivalent to check
	dx, dy, dz, r = get_r3(ex3, ey3, ez3, cx3, cy3, cz3)
	e2lG = kk3(r, G)
	e2lG *= wq
	# equivalent to target
	dx, dy, dz, r = get_r3(ex3, ey3, ez3, tx,  ty,  tz)
	e2tG = kk3(r,  G)
	e2tG *= wq
	# compute the direct eval
	truth = s2tG.dot(tau)
	# compute the FMM eval
	est = e2tG.dot(np.linalg.solve(e2lG, s2lG.dot(tau)))
	# check the error
	err = np.abs(truth-est).max()
	printit('3D, 2nd Order', name, err)
def test_3d_4O(G, Gp, Gpp, name):
	G = protect(G)
	Gp = protect(Gp)
	Gpp = protect(Gpp)
	# source to target
	dx, dy, dz, r = get_r3(sx,  sy, sz,  tx,  ty, tz)
	s2tG   = kk3(r, G)
	# source to check
	dx, dy, dz, r = get_r3(sx, sy, sz, cx3, cy3, cz3)
	s2lGa  = kk3(r, G)
	s2lGb  = kn3(dx, dy, dz, r, Gp)
	s2lG   = np.row_stack([s2lGa, s2lGb])
	# equivalent to check
	dx, dy, dz, r = get_r3(ex3, ey3, ez3, cx3, cy3, cz3)
	e2lGaa = kk3(r, G)
	e2lGab = kn3(dx, dy, dz, r, Gp)
	e2lGba = nk3(dx, dy, dz, r, Gp)
	e2lGbb = nn3(dx, dy, dz, r, Gp, Gpp)
	e2lGa  = np.row_stack([e2lGaa, e2lGab])
	e2lGb  = np.row_stack([e2lGba, e2lGbb])
	e2lG   = np.column_stack([e2lGa, e2lGb])
	# equivalent to target
	dx, dy, dz, r = get_r3(ex3, ey3, ez3, tx, ty, tz)
	e2tGa  = kk3(r, G)
	e2tGb  = nk3(dx, dy, dz, r, Gp)
	e2tG   = np.column_stack([e2tGa, e2tGb])
	# compute the direct eval
	truth = s2tG.dot(tau)
	# compute the FMM eval
	est = e2tG.dot(np.linalg.solve(e2lG, s2lG.dot(tau)))
	# check the error
	err = np.abs(truth-est).max()
	printit('3D, 4th Order', name, err)

##### Begin Tests- ones that we know should work!

print('\n------ Tests that should work! -----')

# 2D Laplace Kernel, 2nd Order
G = lambda r: np.log(r)
test_2d_2O(G, 'Laplace')

# 2D Biharmonic Kernel, 2nd Order
G = lambda r: -r**2*(np.log(r)-1)
Gp = lambda r: -2*r*(np.log(r)-1) - r
Gpp = lambda r: -2*(np.log(r)-1) - 3
test_2d_4O(G, Gp, Gpp, 'Biharmonic')

# 3D Laplace Kernel, 2nd Order
G = lambda r: 1/r
test_3d_2O(G, 'Laplace')

# 3D Biharmonic Kernel, 4th Order
G = lambda r: r
Gp = lambda r: np.ones_like(r)
Gpp = lambda r: np.zeros_like(r)
test_3d_4O(G, Gp, Gpp, 'Biharmonic')

# 3D Modified Biharmonic Kernel, 4th Order
G = lambda r: (np.exp(1j*r)-1)/r
Gp = lambda r: -(np.exp(1j*r)-1)/r**2 + 1j*np.exp(1j*r)/r
Gpp = lambda r: 2*(np.exp(1j*r)-1)/r**3 - 2*np.exp(1j*r)/r**2 - np.exp(1j*r)/r
test_3d_4O(G, Gp, Gpp, 'Biharmonic')

print('\n------ Unknown Tests -----')

# Naomi's Kernel
H0p = lambda x: H1(x)
H0pp = lambda x: H2(x) + H1(x)/x
Y0p = lambda x: -Y1(x)
Y0pp = lambda x: -0.5*(Y0(x) - Y2(x))

G = lambda r: H0(r) - Y0(r)
Gp = lambda r: H0p(r) - Y0p(r)
Gpp = lambda r: H0pp(r) - Y0pp(r)

test_2d_2O(G, 'Naomi')
test_2d_4O(G, Gp, Gpp, 'Naomi')

# Y0
G = lambda r: Y0(r)
test_2d_2O(G, 'Y0')

# H0
G = lambda r: H0(r)
Gp = lambda r: H0p(r)
Gpp = lambda r: H0pp(r)
test_2d_2O(G, 'H0')
test_2d_4O(G, Gp, Gpp, 'H0')
test_3d_2O(G, 'H0')
test_3d_4O(G, Gp, Gpp, 'H0')

# H1
G = lambda r: H1(r)
Gp = lambda r: H2(r) + H1(r)/r
Gpp = lambda r: (H3(r)*r**2 + 3*r*H2(r))/r**2
test_2d_2O(G, 'H1')
test_2d_4O(G, Gp, Gpp, 'H1')
test_3d_2O(G, 'H1')
test_3d_4O(G, Gp, Gpp, 'H1')














def get_r2(x1, y1, x2, y2):
	dx = x2[:,None] - x1
	dy = y2[:,None] - y1
	r = np.hypot(dx, dy)
	return dx, dy, r
def k1k_2(r, G):
	return G(r)
def k1n_2(dx, dy, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	return Gx*nx2[:,None] + Gy*ny2[:,None]
def k2k_2(r, G):
	return G(r)
def k2n_2(dx, dy, r, Gp):
	Gpr = Gp(r)
	Gx = Gpr*dx/r
	Gy = Gpr*dy/r
	return Gx*nx2[:,None] + Gy*ny2[:,None]

def test_2d_4O(G1, G1p, G2, G2p, name):
	G1 = protect(G1)
	G1p = protect(G1p)
	G2 = protect(G2)
	G2p = protect(G2p)
	# source to target
	dx, dy, r = get_r2(sx,  sy,  tx,  ty)
	s2tG   = k1k_2(r, G1)
	# source to check
	dx, dy, r = get_r2(sx, sy, cx2, cy2)
	s2lGa  = k1k_2(r, G1)
	s2lGb  = k1n_2(dx, dy, r, G1p)
	s2lG   = np.row_stack([s2lGa, s2lGb])
	# equivalent to check
	dx, dy, r = get_r2(ex2, ey2, cx2, cy2)
	e2lGaa = k1k_2(r, G1)
	e2lGab = k1n_2(dx, dy, r, G1p)
	e2lGba = k2k_2(r, G2)
	e2lGbb = k2n_2(dx, dy, r, G2p)
	e2lGa  = np.row_stack([e2lGaa, e2lGab])
	e2lGb  = np.row_stack([e2lGba, e2lGbb])
	e2lG   = np.column_stack([e2lGa, e2lGb])
	# equivalent to target
	dx, dy, r = get_r2(ex2, ey2, tx, ty)
	e2tGa  = k1k_2(r, G1)
	e2tGb  = k2k_2(r, G2)
	e2tG   = np.column_stack([e2tGa, e2tGb])
	# compute the direct eval
	truth = s2tG.dot(tau)
	# compute the FMM eval
	est = e2tG.dot(np.linalg.solve(e2lG, s2lG.dot(tau)))
	# check the error
	err = np.abs(truth-est).max()
	printit('2D, 4th Order', name, err)

G1 =  lambda r: -r**2*(np.log(r)-1)
G1p = lambda r: -2*r*(np.log(r)-1) - r
G2 =  lambda r: np.log(r)
G2p = lambda r: 1.0/r
test_2d_4O(G1, G1p, G2, G2p, 'Biharmonic')



