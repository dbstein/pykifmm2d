import pykifmm2d
import numpy as np
import scipy as sp
import scipy.sparse
import time

csr = sp.sparse.csr_matrix
CSR_ADD = pykifmm2d.misc.mkl_sparse.CSR_ADD

n = 10000

A = csr(np.random.rand(n,n))
B = csr(np.random.rand(n,n))

st = time.time()
C1 = A + B
sp_time = time.time() - st
st = time.time()
C2 = CSR_ADD(A, B)
mkl_time = time.time() - st

print('Scipy time is: {:0.1f}'.format(sp_time*1000))
print('MKL   time is: {:0.1f}'.format(mkl_time*1000))
print('Error is: {:0.1e}'.format(np.abs(C1-C2).max()))


