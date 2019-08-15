import numpy as np

tol = 1e-10
B = np.random.rand(AA.shape[0], 10000)
SV = np.linalg.svd(AA)

sel = SV[1] > tol
nv = np.sum(sel)

SV0 = SV[0][:,:nv]
SV1 = SV[1][:nv]
SV2 = SV[2][:nv]

sv1t = SV1[:,None]
def small_multiply(SV0, sv1t, SV2, B):
	A1 = SV2.dot(B)
	ne.evaluate('A1*sv1t', out=A1)
	return SV0.dot(A1)

O1 = AA.dot(B)
O2 = small_multiply(SV0, sv1t, SV2, B)
%timeit AA.dot(B)
%timeit small_multiply(SV0, sv1t, SV2, B)

import line_profiler
%load_ext line_profiler
%lprun -f small_multiply small_multiply(SV0, sv1t, SV2, B)


for i in range(4):
	for j in range(4):
		B = AA[i*48:(i+1)*48,j*48:(j+1)*48]
		SV = np.linalg.svd(B)
		sel = SV[1] > tol
		nv = np.sum(sel)
		print(nv/48)
