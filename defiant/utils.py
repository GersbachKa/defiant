
from defiant.jitting import jit, jit_method
from defiant.linear_solver import chi2_solve
from defiant.orf_functions import get_orf_function
from defiant.custom_exceptions import *

from warnings import warn
from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom

# Constants --------------------------------------------------------------------
@dataclass(frozen=True)
class _Const:
    """A dataclass to hold the various constants used in defiant."""
    DAY_SEC: float = 86400.0  # Number of seconds in a day
    YEAR_SEC: float = (365.24 * 86400.0)  # Number of seconds in a year
    F_YEAR_HZ: float = 1.0 / (365.24 * 86400.0) # Frequency of 1/year in Hz

    C_MPS: float = 299792458.0  # Speed of light in m/s

    PC_M: float = 3.085677581491367e+16  # Parsec in meters
    KPC_M: float = 3.085677581491367e+16 * 1e3  # Kiloparsec in meters
    MPC_M: float = 3.085677581491367e+16 * 1e6  # Megaparsec in meters

constants = _Const() # A single instance of the constants
# Get the values using constants.DAY_SEC, constants.YEAR_SEC, etc.

# Can't JIT this
def binned_os_pair_correlations(xi_ab, rho_ab, sig_ab, bins=10, scheme='pairs'):
    """Compute the binned pair correlations for the OS without pair correlations

    This function computes the binned pair correlations for the OS without pair 
    correlations such that C_ab is diagonal. This function lets users specify
    which binning scheme they want to use. 
    The two options for `scheme` are:
        - 'pairs': Equal number of pairs in each bin.
        - 'separation': Equal separation bins between 0 and pi.

    Alternatively, users can provide an array of bin edges to `bins` which represent
    the edges of the separation bins. In this case, `scheme` is ignored.

    NOTE: This function is not JIT compiled

    Parameters
    ----------
    xi_ab : array
        The array of pulsar pair separations (xi) for each pair. [n_pairs]
    rho_ab : array
        The array of pulsar pair correlations (rho) for each pair. [n_pairs]
    sig_ab : array
        The array of uncertainties (sigma) for each pair. [n_pairs]
    bins : int or array
        The number of bins to use, by default 10. Alternatively, this can be 
        an array of bin edges.
    scheme : str, optional
        The binning scheme to use, by default 'pairs'. Ignored if `bins` is an array.

    Returns
    -------
    tuple
        A tuple containing:
        - xi_avg: The average pulsar pair separation for each bin. [bins]
        - rho_avg: The average correlation for each bin. [bins]
        - sig_avg: The uncertainty of the average correlation for each bin. [bins]

    Raises
    ------
    ValueError
        If the binning scheme is invalid or if the shapes of the input arrays do not match.
    """
    isort = jnp.argsort(xi_ab) # Sort by pulsar pair separation
    xi = xi_ab[isort]
    rho = rho_ab[isort]
    sig = sig_ab[isort]

    # bins could be a number or an array of edges
    if isinstance(bins, int): # 
        # Scheme can be 'pairs', or 'separation'
        if scheme == 'pairs':
            # Equal number of pairs in each bin
            pairs_per_bin = len(xi) // bins
            bin_idx = jnp.arange(0,len(xi)) // pairs_per_bin # Index of bin for each pair
            nbins = bins
        elif scheme == 'separation':
            # Equal separation bins
            edges = jnp.linspace(0, jnp.pi, bins+1) # Edges of separation bins
            bin_idx = jnp.digitize(xi, edges) - 1 # Index of bin for each pair
            nbins = bins
        else:
            raise ValueError(f"Invalid binning scheme: {scheme}")
    else:
        # Assume bins is an array of edges
        bin_idx = jnp.digitize(xi, bins) - 1 # Index of bin for each pair
        nbins = len(bins) - 1

    xi_avg, rho_avg, sig_avg = jnp.zeros(nbins), jnp.zeros(nbins), jnp.zeros(nbins)
    for i in range(nbins):
        # Mask for pairs in this bin
        mask = bin_idx == i
        sub_xi = xi[mask]
        sub_rho = rho[mask]
        sub_sig = sig[mask]

        # Weighted average using inverse variance weighting
        weights = (1 / sub_sig**2)
        xi_avg  = xi_avg.at[i].set(  jnp.average(sub_xi, weights=weights) )
        rho_avg = rho_avg.at[i].set( jnp.average(sub_rho, weights=weights) )
        sig_avg = sig_avg.at[i].set( 1 / jnp.sqrt(jnp.sum(weights)) )
    
    return xi_avg, rho_avg, sig_avg

# Can't JIT this
def binned_pcos_pair_correlations(xi_ab, rho_ab, C_ab, orf='hd', bins=10, scheme='pairs'):
    """Compute the binned pair correlations for the OS with pair correlations

    This function computes the binned pair correlations for the OS with pair 
    correlations such that C_ab is not diagonal. This function lets users specify
    which binning scheme they want to use.

    The two options for `scheme` are:
        - 'pairs': Equal number of pairs in each bin.
        - 'separation': Equal separation bins between 0 and pi.

    Alternatively, users can provide an array of bin edges to `bins` which represent 
    the edges of the separation bins. In this case, `scheme` is ignored.

    `orf` specifies the overlap reduction function to use for the fit. This must
    be a function defined in defiant.orf_functions. The default is 'hd' for the 
    Hellings-Downs curve.

    NOTE: This function is not JIT compiled

    CAUTION: There is a bug in this function where the pulsar separation value (xi_avg)
    is not properly noise-weighted. With PTAs with some pulsars with much higher noise
    than others, this can slightly skew results. This is not a large effect, but
    is something to know.

    Parameters
    ----------
    xi_ab : array
        An array of pulsar pair separations.
    rho_ab : array
        An array of pair correlations.
    C_ab : array
        A covariance matrix for the pair correlations.
    orf : str
        The overlap reduction function to use for the fit, by default 'hd'.
    bins : int
        The number of bins to use, by default 10. Alternatively, this can be 
        an array of bin edges.
    scheme : str, optional
        The binning scheme to use, by default 'pairs'

    Returns
    -------
    tuple
        A tuple containing:
        - xi_avg: The average pulsar pair separation for each bin. [bins]
        - rho_avg: The average correlation for each bin. [bins]
        - sig_avg: The uncertainty of the average correlation for each bin. [bins]

    Raises
    ------
    ValueError
        If the binning scheme is invalid or if the shapes of the input arrays do not match.
    """
    isort = jnp.argsort(xi_ab) # Sort by pulsar pair separation
    xi = xi_ab[isort]
    rho = rho_ab[isort]
    C = C_ab[isort,:][:,isort] # Sort each axis one after the other

    # bins could be a number or an array of edges
    if isinstance(bins, int): # 
        # Scheme can be 'pairs', or 'separation'
        if scheme == 'pairs':
            # Equal number of pairs in each bin
            pairs_per_bin = len(xi) // bins
            bin_idx = jnp.arange(0,len(xi)) // pairs_per_bin # Index of bin for each pair
            nbins = bins
        elif scheme == 'separation':
            # Equal separation bins
            edges = jnp.linspace(0, jnp.pi, bins+1) # Edges of separation bins
            bin_idx = jnp.digitize(xi, edges) - 1 # Index of bin for each pair
            nbins = bins
        else:
            raise ValueError(f"Invalid binning scheme: {scheme}")
    else:
        # Assume bins is an array of edges
        bin_idx = jnp.digitize(xi, bins) - 1 # Index of bin for each pair
        nbins = len(bins) - 1 # Ignore last edge

    xi_avg, rho_avg, sig_avg = jnp.zeros(nbins), jnp.zeros(nbins), jnp.zeros(nbins)
    for i in range(nbins):
        # Mask for pairs in this bin
        mask = bin_idx == i
        sub_xi = xi[mask]
        sub_rho = rho[mask]
        sub_C = C[mask,:][:,mask]

        # Get a sub-design matrix for this bin
        sub_D = get_orf_function(orf)(sub_xi)[:,None] # [N_bin, 1]

        # Solve the chi2 problem for this bin
        A2_bin, Var_A2_bin = chi2_solve(sub_D, sub_C, sub_rho) # [1], [1,1]
        A2_bin, Var_A2_bin = A2_bin[0], Var_A2_bin[0,0] # Get scalars from arrays

        # TODO: Fix this bug. Technically, the separation should be weighted the
        # same way as the correlation.
        bin_xi = jnp.average(sub_xi) # Should be noise weighted...
        bin_orf = jnp.average(sub_D) # Should be noise weighted...

        xi_avg = xi_avg.at[i].set( bin_xi )
        rho_avg = rho_avg.at[i].set( A2_bin * bin_orf )
        sig_avg = sig_avg.at[i].set( jnp.abs(bin_orf) * jnp.sqrt(Var_A2_bin) )    
    
    return xi_avg, rho_avg, sig_avg

# Can't JIT this
def calculate_mcos_average_correlation(A2, Var_A2, orfs=['hd','dp','mp'], n_xi=1000, 
                                       n_samples=1000, eig_thresh=1e-10):
    """Calculate the average correlation function and standard deviation from an MCOS fit

    This function uses numerical sampling to estimate the average correlation function
    as a function of pulsar separation (xi) and the uncertainty on that average 
    correlation function. This function creates many realizations of the `A2` parameters
    using the covariance matrix `Var_A2` and then projects those realizations into the
    correlation space before averaging over the realizations.

    Since this function relies on the covariance matrix `Var_A2`, it is often 
    needed to stabilize that covariance matrix before sampling, done using
    `stabilize_covariance()`. The `eig_thresh` parameter controls the threshold for
    eigenvalues in the stabilization process.

    NOTE: This function is not JIT compiled

    Parameters
    ----------
    A2 : _type_
        _description_
    Var_A2 : _type_
        _description_
    orfs : list, optional
        _description_, by default ['hd','dp','mp']
    n_xi : int, optional
        _description_, by default 1000
    n_samples : int, optional
        _description_, by default 1000
    eig_thresh : _type_, optional
        _description_, by default 1e-10

    Returns
    -------
    tuple
        A tuple containing:
        - xi: The pulsar separation values.
        - avg: The average correlation at each separation.
        - std: The standard deviation of the correlation function at each separation.

    Raises
    ------
    ValueError
        If the shapes of A2, Var_A2, or orfs do not match.
    """
    
    key = get_PRNGKey()
    norf = len(orfs)

    xi = jnp.linspace(0, jnp.pi, n_xi+2)[1:-1] # [n_xi]
    orfs = jnp.array([get_orf_function(orf)(xi) for orf in orfs]) # [n_orf, n_xi]

    if norf != A2.shape[0] or norf != Var_A2.shape[0]:
        msg = f"Mismatched shapes for A2, Var_A2, and number of ORFs"
        raise ValueError(msg)

    # Transform a unit-variance into new space
    u = jrandom.normal(key, shape=(norf, n_samples)) # [n_orf, n_samples]

    # Stabilize the covariance matrix
    C = stabilize_covariance(Var_A2, eig_thresh=eig_thresh) # [n_orf, n_orf]
    # Take cholesky decomposition of covariance matrix
    L = jsl.cholesky(C) # [n_orf, n_orf]

    # Project into the new space
    A2_real = A2[:,None] +  L @ u # [n_orf, n_samples] 

    # Create many realizations of the correlation
    # (n_xi, n_orf) @ (n_orf, n_samples) -> [n_xi, n_samples]
    orf_realizations = orfs.T @ A2_real # [n_xi, n_samples]

    # Calculate mean and sigma over samples
    avg = jnp.mean(orf_realizations, axis=1) # [n_xi]
    std = jnp.std(orf_realizations, axis=1) # [n_xi]

    return xi, avg, std

# Can't JIT this
def get_max_like_params(lfcore, ignore_burn=True):
    """Get the maximum likelihood parameters from a la forge core object

    This function finds the maximum likelihood parameters from a la forge core
    and returns the parameters and index in the chain. If ignore_burn is True,
    it will only consider the samples after the burn-in period.

    NOTE: This function CANNOT be JIT compiled

    Parameters
    ----------
    lfcore : laforge.core.Core
         A la forge core object containing the MCMC chain and parameter names.
    ignore_burn : bool, optional
         A flag to ignore the burn-in period, by default True.

    Returns
    -------
    tuple
        A tuple containing:
        - params: A dictionary of the maximum likelihood parameters.
        - max_i: The corresponding index in the MCMC chain.
    """
    # Not JIT compiled
    burn = lfcore.burn
    lnlike = lfcore.chain[burn:,lfcore.params.index('lnlike')]
    max_i = burn + jnp.argmax(lnlike)
    values = lfcore.chain[max_i]
    params = {p:v for p,v in zip(lfcore.params,values)}

    return params, max_i

# Can't JIT this
def get_freespec_covariance(lfcore, gwb_name):
    """Get the numerical parameter covariance matrix for a freespec GWB 

    This function estimates the covariance matrix of the log10_rho parameters for 
    a freespec GWB signal in an la forge core. This uses jnp.cov() on the MCMC
    chain.

    NOTE: This function CANNOT be JIT compiled

    Parameters
    ----------
    lfcore : laforge.core.Core
         A la forge core object containing the MCMC chain and parameter names.
    gwb_name : str
         The name of the GWB signal in the la forge core.

    Returns
    -------
    jax array
         The covariance matrix of the log10_rho parameters. [nfreqs x nfreqs]
    """
    # Not JIT compiled
    nfreqs = sum(1 for f in lfcore.params if gwb_name+'_log10_rho_' in f)
    chain = [lfcore.get_param(gwb_name+f'_log10_rho_{i}') for i in range(nfreqs)]
    chain = jnp.array(chain) # [nfreqs x N_samples]
    cov = jnp.cov(chain) # [nfreqs x nfreqs]
    return cov

@jit
def compute_pulsar_pair_separations(pos, pair_idx=None):
    """Compute the pulsar pair separations (xi)

    This function computes the pulsar pair separations (xi) for a given list of 
    pulsars and an optional list of pair indexes. If the pair_idx is not provided, 
    it will compute the separations for all unique pairs of pulsars. This function
    assums the pulsar objects have a 'pos' attribute that gives their position 
    in a normalized Cartesian coordinate system.
    
    If pair_idx is None -> pair_idx = jnp.array(jnp.triu_indices(npsrs,k=1)).T

    NOTE: This function is JIT compiled

    Parameters
    ----------
    pos : array
        An array of pulsar positions [npsrs, 3].
    pair_idx : array, optional
        An array of pair indexes [n_pairs, 2], by default None

    Returns
    -------
    jax array
         An array of the pulsar pair separations (xi)
    """
    npsrs = pos.shape[0]

    if pair_idx is None:
        pair_idx = jnp.array(jnp.triu_indices(npsrs,k=1)).T # [N_pairs x 2]

    a,b = pair_idx[:,0], pair_idx[:,1] # [N_pairs], [N_pairs]
    dot = jnp.sum(pos[a,:]*pos[b,:],axis=1) # [N_pairs]
    xi = jnp.arccos(dot) # [N_pairs]
    return xi

@jit(static_argnums=3)
def powerlaw_psd(fgw, log10_A=0.0, gamma=13./3., modes=1):
    """Create a powerlaw PSD for given frequencies and parameters

    This function creates a powerlaw PSD for a given set of frequencies, amplitude,
    and spectral index. The PSD is calculated using the formula:
    P(f) = A^2 / (12 * pi^2 * f_year^3) * (f / f_year)^(-gamma) * df
    where A is the amplitude, f_year is the frequency of 1/year in Hz, 
    gamma is the spectral index, and df is the frequency bin width. The PSD is 
    then repeated for the number of modes. This means that if modes=2, the returned
    PSD_1 = PSD_2. Note that the units of the returned PSD are in s^2

    NOTE: This function is JIT compiled

    Parameters
    ----------
    fgw : array
         An array of frequencies at which to calculate the PSD
    log10_A : float
        The log10 of the amplitude, by default 0.0 -> A=1.0
    gamma : float
        The spectral index, by default 13./3.
    modes : int, optional
        The number of modes to repeat the PSD, by default 1

    Returns
    -------
    jax array
         An array of the powerlaw PSD at the given frequencies, repeated for the 
         number of modes. [nfreqs * modes]
         
    """
    # JIT compiled
    # Calculate frequency bin widths. Bin 0 starts at f=0 and is up to f[0]
    df = jnp.diff(fgw, prepend=0)

    # Calculate the powerlaw (PSD)
    pl = 10**(2*log10_A) * (1/(12*jnp.pi**2 * constants.F_YEAR_HZ**3)) \
         * (fgw/constants.F_YEAR_HZ)**(-gamma)*df

    # Repeat for number of modes
    psd = jnp.repeat(pl,modes)
    return psd

@jit
def matrix_product_trace(X,Y):
    """Calculate the trace of the product of two matrices X and Y

    This function calculates the trace of the product of two matrices X and Y using
    Einstein summation convention. This is equivalent to np.trace(X @ Y) but is
    more memory efficient. This function also works for batches of matrices like [..., N, N].

    NOTE: This function is JIT compiled

    Parameters
    ----------
    X : array
        The first matrix or batch of matrices. [..., N, N]
    Y : array
        The second matrix or batch of matrices. [..., N, N]

    Returns
    -------
    array
        The trace of the product of the two matrices. [...]
    """
    return jnp.einsum('...ij,...ji->...',X,Y)

@jit
def stabilize_covariance(cov, eig_thresh=1e-10):
    """Stabilize a covariance matrix by setting a minimum eigenvalue

    This function takes a covariance matrix, or batch of covariance matrices, 
    and sets the eigenvalues below a certain threshold to that threshold. This 
    is useful for numerical stabilization for things like inversion. This function
    also works for batches of covariance matrices like [..., N, N].

    NOTE: This function is not JIT compiled

    Parameters
    ----------
    cov : array
        The input covariance matrix or batch of covariance matrices. [..., N, N]
    eig_thresh : float
        The threshold for eigenvalues, by default 1e-10

    Returns
    -------
    jax array
        The stabilized covariance matrix or batch of covariance matrices. [..., N, N]
    """
    N = cov.shape[-1]
    idx = jnp.arange(N)
    dims = cov.shape[:-2]

    # Compute eigenvalues and eigenvectors
    e_vals, e_vec = jnp.linalg.eigh(cov) # [dims, N], [dims, N, N]
    # Determine the threshold for eigenvalues
    e_thr = eig_thresh * jnp.max(e_vals, axis=-1, keepdims=True) # [dims, 1]
    # Compute new eigenvalues by thresholding
    e_vals = jnp.where(e_vals < e_thr, e_thr, e_vals) # [dims, N]
    # Construct the diagonal matrix of eigenvalues
    e_mat = jnp.zeros((*dims, N, N))
    e_mat = e_mat.at[..., idx, idx].set(e_vals) # [dims, N, N]
    # Reconstruct the covariance matrix with the modified eigenvalues
    new_cov = e_vec @ e_mat @ jnp.linalg.inv(e_vec) # [dims, N, N]
    return new_cov

@jit
def is_diagonal(C):
    """Check if a matrix is diagonal

    This function checks if the input matrix `C` is diagonal by comparing the 
    sum of all elements to the trace. This function also works for batches of matrices
    like [..., N, N].

    NOTE: This function is JIT compiled

    Parameters
    ----------
    C : array
        The input matrix or batch of matrices. [..., N, N]

    Returns
    -------
    jax array
        Boolean array indicating if each matrix is diagonal [...,]

    Raises
    ------
    ValueError
        If the input matrix is not at least 2D or if the matrices are not square
    """
    # JIT compiled
    if C.ndim < 2 or C.shape[-2] != C.shape[-1]:
        raise ValueError("Input must be at least 2D and square")


    sum_c = jnp.sum(C, axis=(-2,-1)) # Only sum last two axes
    trace_c = jnp.trace(C, axis1=-2, axis2=-1) # Only trace last two axes
    diff = sum_c - trace_c
    is_diag = diff == 0

    return is_diag

# Can't JIT this
def is_nan_or_inf(A):
    """Check if an array contains any NaN or inf values

    This simple function checks if there are any NaN or inf values in the input array.

    Parameters
    ----------
    A : array
         The input array.

    Returns
    -------
    bool
        True if the array contains any NaN or inf values, False otherwise.
    """
    return jnp.isnan(A).any() or jnp.isinf(A).any()

# Can't JIT this
def get_PRNGKey(n=1):
    """Get a random PRNGKey for jax.random function.

    This function generates a random seed using Python's built-in random module 
    and then creates a JAX PRNGKey from that seed. This is useful for starting 
    the random number generation.

    NOTE: This function CANNOT be JIT compiled

    Parameters
    ----------
    n : int, optional
        The number of PRNGKeys to generate, by default 1

    Returns
    -------
    jax.random.PRNGKey
        A single PRNGKey if n=1, or a list of PRNGKeys if n>1
    """
    # Not JIT compiled
    import random
    seed = random.randint(0, 2**32 - 1) 
    keys = jrandom.split(jrandom.PRNGKey(seed), n)
    return keys[0] if n==1 else keys[:n]

# JAX 64 bit checking and enabling ---------------------------------------------
# Can't JIT this
def enable_jax_64bit():
    """Enable JAX 64-bit mode for better numerical stability

    Raises
    ------
    RuntimeError
        If JAX fails to enable 64-bit mode
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Failed to enable JAX 64-bit mode.")

# Can't JIT this
def check_jax_64bit(warn=True):
    """Check if JAX is in 64-bit mode for better numerical stability

    NOTE: This function warns the user if JAX is in 32-bit mode.

    Returns
    -------
    bool
        True if JAX is in 64-bit mode, False otherwise
    """
    import jax
    state = jax.config.jax_enable_x64
    if not state and warn:
        warn("JAX is in 32-bit mode. Enable 64-bit mode for better numerical stability.")
    return state

