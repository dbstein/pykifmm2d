import numpy as np
import numba
import time

def add(x, y, z):
	for i in numba.prange(x.shape[0]):
		z[i] = np.sin(x[i]) + np.cos(y[i])

N = 1000000
x = np.random.rand(N)
y = np.random.rand(N)
z1 = np.empty(N)
z2 = np.empty(N)
z3 = np.empty(N)
z4 = np.empty(N)
z5 = np.empty(N)

numba_add_serial3 = numba.njit(add, cache=True, parallel=True, fastmath=True)
numba_add_parallel3 = numba.njit(add, cache=True, parallel=True)

@numba.njit(parallel=False, cache=True, fastmath=True)
def numba_add_serial4(x, y, z):
	for i in numba.prange(x.shape[0]):
		z[i] = np.sin(x[i]) + np.cos(y[i])

@numba.njit(parallel=True, cache=True, fastmath=True)
def numba_add_parallel4(x, y, z):
	for i in numba.prange(x.shape[0]):
		z[i] = np.sin(x[i]) + np.cos(y[i])

print('\n\n------------------------')

st = time.time()
numba_add_serial3(x, y, z1)
print('Time numba, serial1 (ms):   {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_serial4(x, y, z2)
print('Time numba, serial2 (ms):   {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_parallel3(x, y, z3)
print('Time numba, parallel1 (ms): {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_parallel4(x, y, z4)
print('Time numba, parallel2 (ms): {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
np.sin(x, x)
np.cos(y, y)
np.add(x, y, out=z5)
print('Time numpy (ms):            {:0.1f}'.format((time.time()-st)*1000))

print('------------------------')

st = time.time()
numba_add_serial3(x, y, z1)
print('Time numba, serial1 (ms):   {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_serial4(x, y, z2)
print('Time numba, serial2 (ms):   {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_parallel3(x, y, z3)
print('Time numba, parallel1 (ms): {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
numba_add_parallel4(x, y, z4)
print('Time numba, parallel2 (ms): {:0.1f}'.format((time.time()-st)*1000))

st = time.time()
np.sin(x, x)
np.cos(y, y)
np.add(x, y, out=z5)
print('Time numpy (ms):            {:0.1f}'.format((time.time()-st)*1000))

print('------------------------')

print(np.allclose(z1, z2))
print(np.allclose(z1, z3))
print(np.allclose(z1, z4))
print(np.allclose(z1, z5))

print('------------------------')
