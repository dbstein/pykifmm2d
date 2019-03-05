import numpy as np
import scipy.sparse as sparse
import ctypes
from ctypes import pointer, POINTER, c_void_p, c_int, c_char, c_double
from ctypes import byref, cdll
try:
    try:
      mkl = cdll.LoadLibrary("libmkl_rt.so")
    except:
      mkl = cdll.LoadLibrary("libmkl_rt.dylib")
    mkl_is_here = True
except:
    mkl_is_here = False

if mkl_is_here:
    def SpMV_viaMKL( A, x ):
        """
        Wrapper to Intel's SpMV
        (Sparse Matrix-Vector multiply)
        For medium-sized matrices, this is 4x faster
        than scipy's default implementation
        Stephen Becker, April 24 2014
        stephen.beckr@gmail.com
        """
        SpMV = mkl.mkl_cspblas_dcsrgemv
        # Dissecting the "cspblas_dcsrgemv" name:
        # "c" - for "c-blas" like interface (as opposed to fortran)
        #    Also means expects sparse arrays to use 0-based indexing, which python does
        # "sp"  for sparse
        # "d"   for double-precision
        # "csr" for compressed row format
        # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
        # "mv"  for "matrix-vector" multiply

        if not sparse.isspmatrix_csr(A):
            raise Exception("Matrix must be in csr format")
        (m,n) = A.shape

        # The data of the matrix
        data    = A.data.ctypes.data_as(POINTER(c_double))
        indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
        indices = A.indices.ctypes.data_as(POINTER(c_int))

        # Allocate output, using same conventions as input
        nVectors = 1
        if x.ndim is 1:
           y = np.empty(m,dtype=np.double,order='F')
           if x.size != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
        elif x.shape[1] is 1:
           y = np.empty((m,1),dtype=np.double,order='F')
           if x.shape[0] != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
        else:
           nVectors = x.shape[1]
           y = np.empty((m,nVectors),dtype=np.double,order='F')
           if x.shape[0] != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

        # Check input
        if x.dtype.type is not np.double:
           x = x.astype(np.double,copy=True)
        # Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
        if x.flags['F_CONTIGUOUS'] is not True:
           x = x.copy(order='F')

        if nVectors == 1:
           np_x = x.ctypes.data_as(POINTER(c_double))
           np_y = y.ctypes.data_as(POINTER(c_double))
           # now call MKL. This returns the answer in np_y, which links to y
           SpMV(byref(c_char(b"N")), byref(c_int(m)),data ,indptr, indices, np_x, np_y ) 
        else:
           for columns in range(nVectors):
               xx = x[:,columns]
               yy = y[:,columns]
               np_x = xx.ctypes.data_as(POINTER(c_double))
               np_y = yy.ctypes.data_as(POINTER(c_double))
               SpMV(byref(c_char(b"N")), byref(c_int(m)),data,indptr, indices, np_x, np_y ) 

        return y
    def CSR_ADD( A, B ):
        """
        Wrapper to Intel's mkl_dcsradd
        (Sparse Matrix Addition for CSR MATS)
        """
        # sort indices
        if not A.has_sorted_indices:
          A.sort_indices()
        if not B.has_sorted_indices:
          B.sort_indices

        CSRA = mkl.mkl_dcsradd

        shift = 1

        # variables relating to a, b, beta
        na = A.count_nonzero()
        sha = A.shape
        m = pointer(c_int(sha[0]))
        n = pointer(c_int(sha[1]))
        a = A.data.ctypes.data_as(POINTER(c_double))
        _ja = (A.indices + shift).astype(np.int32)
        ja = _ja.ctypes.data_as(POINTER(c_int))
        _ia = (A.indptr + shift).astype(np.int32)
        ia = _ia.ctypes.data_as(POINTER(c_int))

        beta = byref(c_double(1.0))

        b = B.data.ctypes.data_as(POINTER(c_double))
        _jb = (B.indices + shift).astype(np.int32)
        jb = _jb.ctypes.data_as(POINTER(c_int))
        _ib = (B.indptr + shift).astype(np.int32)
        ib = _ib.ctypes.data_as(POINTER(c_int))
        nzmax = byref(c_int(0))

        # dummy output variables for first call
        c = np.empty(1, dtype=np.float64)
        jc = np.empty(1, dtype=np.int32)
        ic = np.empty(sha[0] + 1, dtype=np.int32)
        pc = c.ctypes.data_as(POINTER(c_double))
        pjc = jc.ctypes.data_as(POINTER(c_int))
        pic = ic.ctypes.data_as(POINTER(c_int))

        # setup variables for MKL call
        trans = pointer(c_char(b"N"))
        request = pointer(c_int(1))
        sort = pointer(c_int(3))
        info = pointer(c_int(0))

        # call once to compute number of values in the ouput
        CSRA(trans, request, sort, m, n, a, ja, ia, beta, b, jb, ib, \
                  pc, pjc, pic, nzmax, info)

        # allocate memory
        nc = pic[m[0]] - 1
        c = np.empty(nc, dtype=np.float64)
        jc = np.empty(nc, dtype=np.int32)
        pc = c.ctypes.data_as(POINTER(c_double))
        pjc = jc.ctypes.data_as(POINTER(c_int))
        request = pointer(c_int(2))
        sort = pointer(c_int(0))
        info = pointer(c_int(0))

        # call once more to compute sum
        CSRA(trans, request, sort, m, n, a, ja, ia, beta, b, jb, ib, \
                  pc, pjc, pic, nzmax, info)

        # construct matrix from the data
        return sparse.csr_matrix((c, jc-shift, ic-shift), shape=sha)

else:
    def SpMV_viaMKL( A, x ):
        return A.dot(x)
    def CSR_ADD(A, B):
        return A + B
