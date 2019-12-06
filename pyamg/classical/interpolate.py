"""Classical AMG Interpolation methods."""


import numpy as np
from numpy.linalg import lstsq
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr, eye
from pyamg import amg_core

__all__ = ['direct_interpolation', 'ls_interpolation']


def direct_interpolation(A, C, splitting):
    """Create prolongator using direct interpolation.

    Parameters
    ----------
    A : csr_matrix
        NxN matrix in CSR format
    C : csr_matrix
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    P : csr_matrix
        Prolongator using direct interpolation

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import direct_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = direct_interpolation(A, A, splitting)
    >>> print P.toarray()
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]

    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C = C.copy()
    C.data[:] = 1.0
    C = C.multiply(A)

    Pp = np.empty_like(A.indptr)

    amg_core.rs_direct_interpolation_pass1(A.shape[0],
                                           C.indptr, C.indices, splitting, Pp)

    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0],
                                           A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting,
                                           Pp, Pj, Px)

    return csr_matrix((Px, Pj, Pp))

def ls_interpolation(A, C, splitting, V, T = None):
    """Create prolongator using least squares interpolation

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N
    V : {array like}
        Test vector(s) for which the interpolation error
        is being minimized in a least-squares sense    
    T : {csr_matrix}
        NfxN matrix in CSR format - composition of all
        previous interpolation operators

    Returns
    -------
    P : {csr_matrix}
        Prolongator using least squares interpolation

    References
    ----------
    [1] Brandt, Achi, James J. Brannick, Karsten Kahl,
    and Irene Livshits. "Bootstrap AMG." SIAM Journal 
    on Scientific Computing 33 (2011): 612-632.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import ls_interpolation
    >>> import numpy as np
    >>> num_tvs = 2
    >>> A = poisson((5,),format='csr')
    >>> V = np.random.random((5, num_tvs))
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = ls_interpolation(A, A, splitting, V)
    """

    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')
    
    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    if (V.shape[0] != A.shape[0]):
        raise ValueError('A and V must have same first dimension')

    # Define T as identity if first level
    if T is None:
        T = eye(A.shape[0]) 
    else:
        if not isspmatrix_csr(T):
            raise TypeError('expected csr_matrix for T')

        if (T.shape[1] != V.shape[0]):
            raise ValueError('T and V must have matching second and first dimensions')

    # 4.1 - Test vector weights are chosen to reflect the energy of the
    #       test vectors in the A-norm and throughout entire hierarchy
    TV = T * V
    AV = A * V
    W = (TV.T * V) / (AV.T * V) # Make sure this does element-wise division
    
    # Interpolation weights are computed based on least squares solution,
    # but subject to the sparsity pattern of C.
    C = C.copy()
    C.data[:] = 1.0

    Pp = np.empty_like(A.indptr)

    # Use strength of connection matrix and C/F splitting to compute row pointer
    # for tentative prolongator
    amg_core.rs_direct_interpolation_pass1(A.shape[0],
                                           C.indptr, C.indices, splitting, Pp)

    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)

    # Calling this to get the column indices for tentative prolongator
    # Not using nonzero entries
    amg_core.rs_direct_interpolation_pass2(A.shape[0],
                                           A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting,
                                           Pp, Pj, Px)

    # Use numpy least squares to fill in tentative prolongator nonzeros 
    W_roots = eye(W.shape[0]) * np.sqrt(W) # Take squareroots of W
    # Loop over rows in prolongator
    for i in range(A.shape[0]):
        start = Pp[i]
        end = Pp[i+1]
        # Sum V_Ci and get V_i
        V_i = V[i:]
        V_Ci = V[Pj[start:end],:] 
        # Calculate least squares solution for row of P and store
        Px[start:end] = lstsq((V_Ci * W_roots).T, V_i * W_roots)

    return csr_matrix((Px, Pj, Pp))
