
from defiant.utils import jit

import jax.numpy as jnp
import jax.scipy.linalg as jsl


def chi2_solve(X, C, r, method=None):
    """Solve the chi-square minimization problem.

    This function solves the chi-square minimization problem for a given
    design matrix X, covariance matrix C, and residuals r. The equation
    (r - X*theta)^T C^(-1) (r - X*theta) is minimized to find the
    solution vector theta: theta = (X.T C^-1 X)^-1 X.T C^-1 r. The method 
    parameter determines the approach used to solve the linear system.

    This method requires the following shapes:
        - X: (N x M)
        - C: (N x N) or (N) for diagonal
        - r: (N x 1)
    
    Available methods are:
        - diagonal: Use exact diagonal inversion when C is shape (N)
        - forward: Use jnp.linalg.solve for forward substitution
        - pinv: Use pseudo matrix inverses (jnp.linalg.pinv)
        - cholesky: Use cholesky decomposition for forward substitution

    The preferred method for diagonal C matrices is 'diagonal', while for dense
    matrices 'forward' is preferred.

    Parameters
    ----------
    X : array
        Design matrix (N x M)
    C : array
        Covariance matrix (N x N) or (N) for diagonal
    r : array
        Residuals (N x 1)
    method : str, optional
        Method to use for solving the linear system. See description for options. 
        By default None

    Returns
    ------- 
    tuple
        A tuple containing:
        - theta (array): The solution vector (M x 1)
        - cov (array): The covariance matrix between linear elements (M x M)

    Raises
    ------
    NameError
        If an unknown method is specified.
    """

    if method is None:
        if C.ndim == 1:
            method = 'diagonal'
        else:
            method = 'forward'
    
    if method.lower() == 'diagonal':
        return _diagonal_solve(X, C, r)
    if method.lower() == 'pinv':
        return _pinv_solve(X, C, r)
    if method.lower() == 'cholesky':
        return _cholesky_solve(X, C, r)
    if method.lower() == 'forward':
        return _forward_solve(X, C, r)
    else:
        msg = f'Unknown method \'{method}\' for linear solving.'
        raise NameError(msg)


@jit
def _diagonal_solve(X, C, r):
    """Solve the linear system using the diagonal method.

    This method assumes that the covariance matrix C is diagonal, which allows
    for a fast, accurate solution. 
    NOTE: C must be provided as the diagonal elements only (N,).

    Parameters
    ----------
    X : array
        Design matrix (N x M)
    C : array
        The diagonal of the covariance matrix (N)
    r : array
        Residuals (N x 1)

    Returns
    -------
    tuple
        A tuple containing:
        - theta (array): The solution vector (M x 1)
        - cov (array): The covariance matrix between linear elements (M x M)
    """
    # r is (N x 1); X is (N x M); C is (N) [just the diagonal elements]
    Cinv = 1/C  # (N)
    CinvX = X * Cinv[:,None]  # (N x M) 

    fisher = CinvX.T @ X   # (M x M)
    dirty_map = CinvX.T @ r   # (M x 1)

    # fisher is a relatively small matrix, use inv here for speed
    cov = jnp.linalg.inv(fisher)  # (M x M)
    theta = cov @ dirty_map  # (M x 1)
    return theta, cov


@jit
def _pinv_solve(X, C, r):
    """Solve the linear system using the pseudo-inverse method.

    This method uses jnp.linalg.pinv to compute the pseudo-inverse of the
    covariance matrix C. It is suitable for ill-conditioned problems where
    the exact inverse may not be stable.

    Parameters
    ----------
    X : array
        Design matrix (N x M)
    C : array
        Covariance matrix (N x N)
    r : array
        Residuals (N x 1)

    Returns
    -------
    tuple
        A tuple containing:
        - theta (array): The solution vector (M x 1)
        - cov (array): The covariance matrix between linear elements (M x M)
    """
    # r is (N x 1); X is (N x M); C is (N x N)
    Cinv = jnp.linalg.pinv(C)  # (N x N)
    CinvX = Cinv @ X # (N x M) 

    fisher = CinvX.T @ X  # (M x M)
    dirty_map = CinvX.T @ r  # (M x 1)

    # fisher is a relatively small matrix, use inv here for speed
    cov = jnp.linalg.inv(fisher)  # (M x M)
    theta = cov @ dirty_map  # (M x 1)
    return theta, cov


@jit
def _cholesky_solve(X, C, r):
    """Solve the linear system using the Cholesky decomposition method.

    This method uses Cholesky decomposition to solve the linear system.
    It is efficient for positive definite matrices. But is slower than the
    forward method.

    Parameters
    ----------
    X : array
        Design matrix (N x M)
    C : array
        Covariance matrix (N x N)
    r : array
        Residuals (N x 1)

    Returns
    -------
    tuple
        A tuple containing:
        - theta (array): The solution vector (M x 1)
        - cov (array): The covariance matrix between linear elements (M x M)
    """
    # r is (N x 1); X is (N x M); C is (N x N)
    cf = jsl.cho_factor(C)  # Cholesky factorization
    CinvX = jsl.cho_solve(cf, X) # (N x M) 

    fisher = CinvX.T @ X  # (M x M)
    dirty_map = CinvX.T @ r  # (M x 1)

    # fisher is a relatively small matrix, use inv here for speed
    cov = jnp.linalg.inv(fisher)  # (M x M)
    theta = cov @ dirty_map  # (M x 1)
    return theta, cov


@jit
def _forward_solve(X, C, r):
    """Solve the linear system using the forward substitution method.

    This method uses forward substitution to solve the linear system.
    It is efficient for lower triangular matrices. This method uses
    jnp.linalg.solve for the forward substitution.

    Parameters
    ----------
    X : array
        Design matrix (N x M)
    C : array
        Covariance matrix (N x N)
    r : array
        Residuals (N x 1)

    Returns
    -------
    tuple
        A tuple containing:
        - theta (array): The solution vector (M x 1)
        - cov (array): The covariance matrix between linear elements (M x M)
    """
    # r is (N x 1); X is (N x M); C is (N x N)
    CinvX = jnp.linalg.solve(C, X)

    fisher = CinvX.T @ X  # (M x M)
    dirty_map = CinvX.T @ r  # (M x 1)

    # fisher is a relatively small matrix, use inv here for speed
    cov = jnp.linalg.inv(fisher)  # (M x M)
    theta = cov @ dirty_map  # (M x 1)
    return theta, cov

