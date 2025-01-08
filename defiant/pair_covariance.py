
from .custom_exceptions import *
from .utils import *

import numpy as np

from itertools import combinations, combinations_with_replacement
from tqdm.auto import tqdm


def _compute_pair_covariance(Z, phi1, phi2, orf, norm_ab, a2_est, use_tqdm, max_chunk):
    """A function to compute the pulsar pair covariance matrix. Not intended for user use.

    This function computes the pulsar pair covariance matrix as described in
    Gersbach et al. 2024. Specifically, this function separates the diagonal entries 
    from the off-diagonals so that we can use the woodbury matrix identity for increase 
    numerical stability. This function uses two forms of phi so as to be agnostic
    to both the OS and PFOS. phi1 is intended to be either \hat{\phi} in the case
    of the OS and \tilde{\phi}(f_k) in the case of the PFOS. phi2 is intended to be
    \hat{\phi} in the case of the OS and \Phi(f_k) in the case of the PFOS. Similarly
    norm_ab represents the normalization of the pair-wise estimators. For the OS,
    norm_ab is simply sig_ab. For the PFOS, norm_ab changes depending on if you
    are using wideband or narrowband normalization. 

    Args:
        Z (np.ndarray): An array of Z matrix products [N_pulsars x 2N_frequencies x 2N_frequencies]
        phi1 (np.ndarray): An array of the spectral model [2N_frequencies] (See description)
        phi2 (np.ndarray): A similar array of the spectral model [2N_frequencies] (See description)
        orf (np.ndarray): A matrix of the ORF for each pair of pulsars [N_pulsars x N_pulsars]
        norm_ab (np.ndarray): The normalization terms for each pulsar pair [N_pairs]
        a2_est (float): The estimated amplitude of the GWB
        use_tqdm (bool): A flag to use the tqdm progress bar
        max_chunk (int): The maximum number of simultaneous matrix calculations

    Returns:
        np.ndarray: The pulsar pair covariance matrix [2, N_pairs x N_pairs]
    """
    fact_c = _factored_pair_covariance(Z, phi1, phi2, orf, norm_ab, 
                                       use_tqdm, max_chunk)
    
    return fact_c[0] + a2_est*fact_c[1] + a2_est**2*fact_c[2]


def _compute_mcos_pair_covariance(Z, phi1, phi2, orf, design, 
                                  rho_ab, sig_ab, norm_ab, a2_est,
                                  use_tqdm, max_chunk):
    """Compute the pulsar pair covariance matrix with the MCOS. Not intended for user use.

    This function computes the pulsar pair covariance matrix with the MCOS as described in
    Sardesai et al. 2023 and Gersbach et al. 2024. As was done in the first paper,
    the pair covariance needs an estimated total correlated power in each pair.
    We do that by first calculating the non-pair covariant MCOS and then using the 
    resulting fraction of power in each process multiplied by the CURN amplitude 
    as an estimate of the total correlated power. This function uses two forms of phi 
    so as to be agnostic to both the OS and PFOS. phi1 is intended to be either 
    \hat{\phi} in the case of the OS and \tilde{\phi}(f_k) in the case of the PFOS. 
    phi2 is intended to be \hat{\phi} in the case of the OS and \Phi(f_k) in the 
    case of the PFOS. Similarly, norm_ab represents the normalization of the 
    pair-wise estimators. For the OS, norm_ab is simply sig_ab. For the PFOS, 
    norm_ab changes depending on if you are using wideband or narrowband normalization.

    Args:
        Z (np.ndarray): An array of Z matrix products [N_pulsars x 2N_frequencies x 2N_frequencies]
        phi1 (np.ndarray): An array of the spectral model [2N_frequencies] (See description)
        phi2 (np.ndarray): A similar array of the spectral model [2N_frequencies] (See description)
        orf (np.ndarray): A matrix of the ORF for each pair of pulsars [N_pulsars x N_pulsars]
        design (np.ndarray): The design matrix for the MCOS [N_pairs x M_orfs]
        rho_ab (np.ndarray): The rho values for each pair of pulsars [N_pairs]
        sig_ab (np.ndarray): The sig values for each pair of pulsars [N_pairs]
        norm_ab (np.ndarray): The normalization terms for each pulsar pair [N_pairs]
        a2_est (float): The estimated amplitude of the GWB
        use_tqdm (bool): A flag to use the tqdm progress bar
        max_chunk (int): The maximum number of simultaneous matrix calculations

    Returns:
        np.ndarray: The pulsar pair covariance matrix [2, N_pairs x N_pairs]
    """
    # Need to compute the a no-PC MCOS for amp estimates
    temp_c = np.square(sig_ab)
    mcos,_ = linear_solve(design,temp_c,rho_ab,'diagonal')

    # Set negative values to 0
    mcos[mcos<0] = 0

    # Normalize the power per process
    norm_pow = mcos/np.sqrt(np.dot(mcos,mcos))
    est_pow = a2_est*norm_pow

    # Hijack the factored code by giving it correlated power in ORF and A2=1!
    cor_pow = np.sum([o*a for o,a in zip(orf,est_pow)], axis=0)

    fact_c = _factored_pair_covariance(Z, phi1, phi2, cor_pow, norm_ab, 
                                       use_tqdm, max_chunk)
    return fact_c[0] + fact_c[1] + fact_c[2]


# Hidden function which directly calculates the factored pair covariance matrix.
# Exposed functions will call this, but users should avoid.
def _factored_pair_covariance(Z, phi1, phi2, orf, norm_ab, use_tqdm, max_chunk):
    """Creates the GW amplitude factored pulsar pair covariance matrix.

    This function uses numpy array indexing shenanigans and numpy vectorized
    operations to compute a factored version of the pulsar pair covariance matrix.
    The format of the returned covariance matrix is a 3 x N_pairs x N_pairs matrix
    where the first term should be multiplied by 1, the second by A^2, and
    the 3rd by A^4 before summing.

    Args:
        Z (numpy.ndarray): A N_pulasr array of 2N_frequencies x 2N_frequencies Z matrices from the OS
        phi1 (numpy.ndarray): A 2N_frequencies array of the estimator's normalized spectral model
        phi2 (numpy.ndarray): A 2N_frequencies array of the unit spectral shape of the GWB
        orf (numpy.ndarray): A N_pulsar x N_pulsar matrix of the ORF (or correlated power) 
                for each pair of pulsars
        norm_ab (numpy.ndarray): The N_pair array of normalizations of the pair-wise estimators 
        use_tqdm (bool): A flag to use the tqdm progress bar
        max_chunk (int): The maximum number of simultaneous matrix calculations. 
                Works best between 100-1000 but depends on the computer. Defaults to 300.

    Raises:
        PCOSInteruptError: If the pair covariance calculation is interupted

    Returns:
        numpy.ndarray: A 3 x N_pairs x N_pairs matrix of the final amplitude factored covariance matrix
    """
    # Use some pre-calculations to speed up processing
    npsr = len(Z)
    nfreq = len(Z[0])

    pairs_idx = np.array(list( combinations(range(npsr),2) ),dtype=int)
    a,b = pairs_idx[:,0], pairs_idx[:,1]

    # Get pairs of pairs, both the indices of the pairs, and the pulsar indices
    PoP_idx = np.array(list( combinations_with_replacement(range(len(pairs_idx)),2) ),dtype=int)
    
    PoP = np.zeros((len(PoP_idx),4),dtype=int)
    PoP[:,(0,1)] = pairs_idx[PoP_idx[:,0]] 
    PoP[:,(2,3)] = pairs_idx[PoP_idx[:,1]]

    # It is also helpful to create some basic filters. From (ab,cd)
    psr_match = (PoP[:,(0,1)] == PoP[:,(2,3)]) # checks (a==c,b==d)
    psr_inv_match = (PoP[:,(0,1)] == PoP[:,(3,2)]) # checks (a==d,b==c)

    # It will be faster to pre-compute some quantities
    Zphi1 = phi1*Z
    Zphi2 = phi2*Z
    Zphi1Zphi2 = np.zeros((npsr,npsr, nfreq,nfreq))

    Zphi1Zphi2[a,b] = Zphi1[a] @ Zphi2[b]
    Zphi1Zphi2[b,a] = Zphi1[b] @ Zphi2[a]

    # Create the progress bar
    progress_bar = tqdm(total=len(PoP),desc='PC elements',leave=False) if use_tqdm else None

    # Define a lambda function for easy reading
    mpt = lambda A,B: matrix_product_trace(A,B)

    # Define the three cases for the pair covariance
    def case1(a,b,c,d):             #(ab,cd)
        a0 = np.zeros_like(a)
        a2 = np.zeros_like(a)
        a4 = orf[a,c]*orf[d,b] * mpt(Zphi1Zphi2[b,a], Zphi1Zphi2[c,d]) + \
             orf[a,d]*orf[c,b] * mpt(Zphi1Zphi2[b,a], Zphi1Zphi2[d,c])
        return [a0,a2,a4]
    
    def case2(a,b,c):               #(ab,ac)
        a0 = np.zeros_like(a)
        a2 = orf[b,c] *          mpt(Zphi1[b],        Zphi1Zphi2[a,c])
        a4 = orf[a,c]*orf[a,b] * mpt(Zphi1Zphi2[b,a], Zphi1Zphi2[c,a])
        return [a0,a2,a4]

    def case3(a,b):                 #(ab,ab)
        a0 =               mpt(Zphi1[b],        Zphi1[a])
        a2 = np.zeros_like(a)
        a4 = orf[a,b]**2 * mpt(Zphi1Zphi2[b,a], Zphi1Zphi2[b,a])
        return [a0,a2,a4]

    # --------------------------------------------------------------------------
    # Chunking code
    def chunker(idx1,idx2,a,b,c=None,d=None):
        n_chunks = int(len(a)/max_chunk)+1
        for i in range(n_chunks):
            l,h = i*max_chunk,(i+1)*max_chunk

            if (c is None) and (d is None):
                temp = case3(a[l:h],b[l:h])
            elif d is None:
                temp = case2(a[l:h],b[l:h],c[l:h])
            else:
                temp = case1(a[l:h],b[l:h],c[l:h],d[l:h])

            C_m[:,idx1[l:h],idx2[l:h]] = temp
            C_m[:,idx2[l:h],idx1[l:h]] = temp

            if use_tqdm: progress_bar.update(h-l)


    # --------------------------------------------------------------------------
    # Now lets calculate them!
    C_m = np.zeros((3,len(pairs_idx),len(pairs_idx)),dtype=np.float64)

    try: # This is used to close the progress bar if an exception occurs

        # Case1: no matching pulsars--------------------------------------------
        mask = (~psr_match[:,0] & ~psr_match[:,1]) & \
               (~psr_inv_match[:,0] & ~psr_inv_match[:,1])
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,c,d = PoP[mask].T

        chunker(p_idx1,p_idx2,a,b,c,d)
    
        
        # Case2: 1 matching pulsar----------------------------------------------
        mask = (psr_match[:,0] & ~psr_match[:,1]) # Check for (ab,ac)
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,c = PoP[mask].T

        chunker(p_idx1,p_idx2,a,b,c)


        mask = (~psr_inv_match[:,0] & psr_inv_match[:,1]) # Check for (ab,bc)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,_,c = PoP[mask].T # Index swap a with b

        chunker(p_idx1,p_idx2,a,b,c)


        mask = (~psr_match[:,0] & psr_match[:,1]) # Check for (ab,cb)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,c,_ = PoP[mask].T # Index swap a with b

        chunker(p_idx1,p_idx2,a,b,c)


        # Case3: 2 matching pulsars---------------------------------------------
        mask = psr_match[:,0] & psr_match[:,1] # Check for (ab,ab)
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,_ = PoP[mask].T 

        chunker(p_idx1,p_idx2,a,b)


    except Exception as e:
        if use_tqdm: progress_bar.close()
        tqdm.close()
        msg = 'Exception occured during pair covariance creation!'
        raise PCOSInteruptError(msg) from e
        
    if use_tqdm: progress_bar.close()

    # Include the final sigmas
    C_m[:] *= np.outer(norm_ab,norm_ab)

    return C_m


def matrix_product_trace(A,B):
    """Calculates the trace of the matrix product of two matrices.

    returns tr(A @ B), supports vectorized operations. Be careful when giving
    large chunks of matrices as it can be memory intensive.

    Args:
        A (numpy.ndarray): A matrix or a stack of n matrices [(n x) M x O]
        B (numpy.ndarray): A matrix or a stack of n matrices [(n x) O x M]

    Raises:
        ValueError: If A and B are not 2D or 3D arrays
    
    Returns:
        float: The trace of the matrix product of A and B
    """
    if A.ndim == 2 and B.ndim == 2:
        return np.einsum('ij,ji->', A, B)
    
    elif A.ndim == 3 and B.ndim == 3:
        return np.einsum('ijk,ikj->i', A, B)

    else:
        raise ValueError('A and B must be 2D or 3D arrays!')