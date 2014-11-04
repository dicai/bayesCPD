cimport numpy as np
import numpy as np
from scipy.special import psi
from libc.math cimport pow

FLOAT = np.float
ctypedef np.float_t FLOAT_t

INT = np.int
ctypedef np.int_t INT_t

DOUBLE = np.double
ctypedef np.double_t DOUBLE_t

#cdef extern from "test_digamma.hpp":
#    float _digamma_cython "digamma_cython"(float x)

modecollapse = lambda r, theta: theta[r.argmax()]
meancollapse = lambda r, theta: theta.mean(0)
constant_hazard = lambda r, rate: np.ones(np.size(r)) / rate


###############################################################################
# Pure Python Functions
###############################################################################

def get_full_row(row, inds, base):

    newrow = np.array([])

    for i in xrange(len(inds) - 1):
        start = inds[i]; end = inds[i+1]
        queue = row[start:end]
        for j in queue:
            newrow = np.append(newrow, np.repeat(j, base ** i) / float(base**i))

    return newrow


###############################################################################
# Typedefed Functions
###############################################################################
def get_full_row_cython(np.ndarray[FLOAT_t] row, np.ndarray[INT_t] inds, int base):

    cdef np.ndarray[FLOAT_t] newrow = np.array([], dtype=FLOAT)
    cdef np.ndarray[FLOAT_t] queue
    cdef int I = len(inds) - 1
    cdef int Q
    cdef int start, end
    cdef float size

    for i in range(I):

        start = inds[i]
        end = inds[i+1]

        queue = row[start:end]
        #Q = len(queue)
        Q = end - start
        size = 1.* base ** i

        for q in range(Q):
            newrow = np.append(newrow, np.ones(size)*queue[q] / size)
            #newrow = np.append(newrow, np.repeat(queue[q], base**i) / float(base**i))

    return newrow

###############################################################################
# Using C++ Functions
###############################################################################

