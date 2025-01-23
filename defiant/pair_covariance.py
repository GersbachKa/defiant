
from .custom_exceptions import *
from .utils import *

import numpy as np
from tqdm.auto import tqdm


def create_OS_pair_covariance(Z, phihat, phi, orf, norm_ab, use_tqdm=True, max_chunk=300):
    """Creates the GWB correlated pair covariance matrix for the OS.

    This function uses numpy array indexing shenanigans and numpy vectorized
    operations to quickly compute the pulsar pair covariance matrix. This function
    is designed for the OS but can be hacked to work with the PFOS.

    Importantly, this function uses the separate phihat and phi matrices which represent
    the unit-amplitude spectral model and the total estimated spectral model respectively.
    Such that phi = A2 * phihat. With this version, you do not need to estimate the ampltidue
    and instead estimate phi (which enterprise can do easily).

    For the traditional OS, norm_ab = sig_ab**2.

    Args:
        Z (numpy.ndarray): A N_pulasr array of 2N_frequencies x 2N_frequencies Z matrices from the OS.
        phihat (numpy.ndarray): A 2N_frequencies array of the unit-amplitude spectral model.
        phi (numpy.ndarray): A 2N_frequencies array of the estimated spectrum.
        orf (numpy.ndarray): A N_pulsar x N_pulsar matrix of the ORF for each pair of pulsars.
        norm_ab (numpy.ndarray): The N_pair array of pair-wise estimator normalizations.
        use_tqdm (bool): A flag to use the tqdm progress bar. Defaults to True.
        max_chunk (int): The maximum number of simultaneous matrix calculations. 
            Works best between 100-1000 but depends on the computer. Defaults to 300.

    Raises:
        PCOSInteruptError: If the pair covariance calculation is interupted

    Returns:
        numpy.ndarray: A N_pairs x N_pairs matrix of the final covariance matrix
    """
    # Use some pre-calculations to speed up processing
    npsr = len(Z)
    nfreq = len(Z[0])//2
    npair = npsr*(npsr-1)//2

    pairs_idx = np.array(np.triu_indices(npsr,1)).T
    a,b = pairs_idx[:,0], pairs_idx[:,1]

    # Get pairs of pairs, both the indices of the pairs, and the pulsar indices
    PoP_idx = np.array(np.triu_indices(npair)).T
    
    PoP = np.zeros((len(PoP_idx),4),dtype=int)
    PoP[:,(0,1)] = pairs_idx[PoP_idx[:,0]] 
    PoP[:,(2,3)] = pairs_idx[PoP_idx[:,1]]

    # It is also helpful to create some basic filters. From (ab,cd)
    psr_match = (PoP[:,(0,1)] == PoP[:,(2,3)]) # checks (a==c,b==d)
    psr_inv_match = (PoP[:,(0,1)] == PoP[:,(3,2)]) # checks (a==d,b==c)

    # It will be faster to pre-compute some quantities
    # For this, we should compute the Z @ phi @ Z @ phihat and Z @ phihat
    ZphiZphihat = np.zeros((npsr,npsr, 2*nfreq,2*nfreq))
    ZphiZphihat[a,b] = phihat*((phi*Z[a]) @ Z[b])
    ZphiZphihat[b,a] = phihat*((phi*Z[b]) @ Z[a])

    Zphihat = phihat*Z

    # Create the progress bar
    progress_bar = tqdm(total=len(PoP),desc='PC elements',leave=False) if use_tqdm else None

    # Define a lambda function for easy reading
    mpt = lambda A,B: matrix_product_trace(A,B)

    # Define the three cases for the pair covariance
    def case1(a,b,c,d):             #(ab,cd)
        # = gamma_{ac} gamma_{bd} tr([Z_d phi Z_b] phihat [Z_a phi Z_c] phihat) + 
        #   gamma_{ad} gamma_{bc} tr([Z_c phi Z_b] phihat [Z_a phi Z_d] phihat)
        a0 = np.zeros_like(a)
        a2 = np.zeros_like(a)
        a4 = orf[a,c]*orf[d,b] * mpt(ZphiZphihat[d,b], ZphiZphihat[a,c]) + \
             orf[a,d]*orf[c,b] * mpt(ZphiZphihat[c,b], ZphiZphihat[a,d])
        return a0+a2+a4
    
    def case2(a,b,c):               #(ab,ac)
        # = gamma_{bc}            tr([Z_c phi Z_b] phihat [Z_a] phihat) + 
        #   gamma_{ac} gamma_{ab} tr([Z_a phi Z_b] phihat [Z_a phi Z_c] phihat)
        a0 = np.zeros_like(a)
        a2 = orf[b,c] *          mpt(ZphiZphihat[c,b],Zphihat[a])
        a4 = orf[a,c]*orf[a,b] * mpt(ZphiZphihat[a,b],ZphiZphihat[a,c])
        return a0+a2+a4

    def case3(a,b):                 #(ab,ab)
        # = tr(Z_b phihat Z_a phihat) + gamma_{ab}^2 tr([Z_a phi Z_b] phihat [Z_a phi Z_b] phihat)
        a0 =               mpt(Zphihat[b],Zphihat[a])
        a2 = np.zeros_like(a)
        a4 = orf[a,b]**2 * mpt(ZphiZphihat[b,a],ZphiZphihat[b,a])
        return a0+a2+a4

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

            C_m[idx1[l:h],idx2[l:h]] = temp
            C_m[idx2[l:h],idx1[l:h]] = temp

            if use_tqdm: progress_bar.update(h-l)


    # --------------------------------------------------------------------------
    # Now lets calculate them!
    C_m = np.zeros((len(pairs_idx),len(pairs_idx)),dtype=np.float64)

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
        msg = 'Exception occured during pair covariance creation!'
        raise PCOSInteruptError(msg) from e
        
    if use_tqdm: progress_bar.close()

    # Include the final sigmas
    C_m *= np.outer(norm_ab,norm_ab)

    return C_m


def create_MCOS_pair_covariance(Z, phihat, orfs, design_mat, rho_ab, sig_ab, a2_est, 
                                use_tqdm=True, max_chunk=300):
    """Compute the pulsar pair covariance matrix with the MCOS.

    This function computes the pulsar pair covariance matrix with the MCOS as described in
    Sardesai et al. 2023. This function is designed purely for the OS and not PFOS.
    
    The method (detailed in Sardesai et al. 2023) is as follows:
    - Calculate the non-pair covariant MCOS
    - Use the resulting fractional power per process to estimate the fractional correlated power
    - Multiply the fractional correlated power by the CURN amplitude to get the total correlated power
    - Use the total correlated power to calculate the pair covariance matrix with phi => phihat
    
    Args:
        Z (np.ndarray): An array of Z matrix products [N_pulsars x 2N_frequencies x 2N_frequencies]
        phihat (numpy.ndarray): An array of the unit-amplitude spectral model [2N_frequencies]
        orfs (np.ndarray): A matrix of the ORF for each pair of pulsars for each ORF [N_orf x N_pulsars x N_pulsars]
        design_mat (np.ndarray): The design matrix for the MCOS [N_pairs x M_orfs]
        rho_ab (np.ndarray): The rho values for each pair of pulsars [N_pairs]
        sig_ab (np.ndarray): The sig values for each pair of pulsars [N_pairs]
        a2_est (float): The estimated amplitude of the GWB
        use_tqdm (bool): A flag to use the tqdm progress bar. Defaults to True.
        max_chunk (int): The maximum number of simultaneous matrix calculations. 
            Works best between 100-1000 but depends on the computer. Defaults to 300.

    Returns:
        np.ndarray: The pulsar pair covariance matrix [2, N_pairs x N_pairs]
    """
    # Need to compute the a no-PC MCOS for amp estimates
    mcos,_ = linear_solve(design_mat, np.diag(sig_ab**2), rho_ab, None, method='diagonal')
    mcos = mcos[:,0]

    # Set negative values to 0
    mcos[mcos<0] = 0

    # Normalize the power per process
    norm_pow = mcos/np.sqrt(np.dot(mcos,mcos))
    est_pow = a2_est*norm_pow

    # Hijack the create_OS_pair_covariance function by supplying correlated power in the ORFs
    # instead of giving the power estimate in phi
    cor_pow = np.sum([o*a for o,a in zip(orfs,est_pow)], axis=0)

    return create_OS_pair_covariance(Z, phihat, phihat, cor_pow, sig_ab**2, use_tqdm, max_chunk)


def create_PFOS_pair_covariance(Z, phi, orf, norm_abk, narrowband, use_tqdm=True, max_chunk=300):
    """Creates the GWB correlated pair covariance matrix for the PFOS.

    This function creates the GWB correlated pair covariance matrix for the PFOS
    for each frequency bin. This function takes advantage of the create_OS_pair_covariance
    function where phihat is replaced with \tilde{\phi}(f_k) for each frequency.

    Args:
        Z (numpy.ndarray): A N_pulasr array of 2N_frequencies x 2N_frequencies Z matrices from the OS.
        phi (numpy.ndarray): A 2N_frequencies array of the estimated spectrum.
        orf (numpy.ndarray): A N_pulsar x N_pulsar matrix of the ORF for each pair of pulsars.
        norm_abk (numpy.ndarray): The N_pair array of PFOS normalizations for each frequency.
        narrowband (bool): A flag to use the narrowband normalization instead of the broadband normalization.
        use_tqdm (bool): A flag to use the tqdm progress bar. Defaults to True.
        max_chunk (int): The maximum number of simultaneous matrix calculations. 
            Works best between 100-1000 but depends on the computer. Defaults to 300.

    Returns:
        numpy.ndarray: A N_freq x N_pairs x N_pairs matrix of the final covariance matrix
    """
    nfreq = len(Z[0])//2
    # The frequency selector matrices (set of diagonals)
    phitilde = np.repeat(np.diag(np.ones(nfreq)),2,axis=1)

    Ck = []
    iterable = tqdm(range(nfreq),desc='Freq covariances') if use_tqdm else range(nfreq)
    for k in iterable:
        if narrowband:
            # Need to include the S(f_k) in the second \tilde{\phi}(f_k) to get the correct units
            C = create_OS_pair_covariance(Z, phitilde[k], phi[2*k]*phitilde[k], orf, norm_abk[k], False, max_chunk)
        else:
            C = create_OS_pair_covariance(Z, phitilde[k], phi, orf, norm_abk[k], False, max_chunk)
        Ck.append(C)
    
    return np.array(Ck)


def create_MCPFOS_pair_covariance(Z, phi, orfs, norm_abk, design_mat, rho_abk, sig_abk,
                                  narrowband, use_tqdm=True, max_chunk=300):
    """Create the GWB correlated pair covariance matrix for the Multi Component PFOS.

    This function creates the GWB correlated pair covariance matrix for the MCOS
    for each frequency bin. This function takes advantage of the create_OS_pair_covariance
    function where phihat is replaced with \tilde{\phi}(f_k) for each frequency.
    This function also utilizes the same strategies found in create_MCOS_pair_covariance
    to estimate the correlated power per process.

    This function computes the pulsar pair covariance matrix with the MCOS as described in
    Sardesai et al. 2023. 
    
    The method (detailed in Sardesai et al. 2023) is as follows:
    - Calculate the non-pair covariant MCPFOS
    - Use the resulting fractional power per process per frequency to estimate the 
        fractional correlated power per frequency
    - Multiply the fractional correlated power per frequency by the CURN PSD to get 
        the total correlated power per frequency
    - Use the total correlated power per frequency to calculate the pair covariance 
        matrix with phi => \tilde{\phi}(f_k)

    Args:
        Z (np.ndarray): An array of Z matrix products [N_pulsars x 2N_frequencies x 2N_frequencies]
        phi (numpy.ndarray): An array of the estimated spectrum [2N_frequencies]
        orfs (np.ndarray): A matrix of the ORF for each pair of pulsars for each ORF [N_orf x N_pulsars x N_pulsars]
        norm_abk (np.ndarray): The N_pair array of PFOS normalizations for each frequency [N_freq x N_pairs]
        design_mat (np.ndarray): The design matrix for the MCOS [N_pairs x M_orfs]
        rho_abk (np.ndarray): The rho_k values for each pair of pulsars for each frequency [N_freq x N_pairs]
        sig_abk (np.ndarray): The sig_k values for each pair of pulsars for each frequency [N_freq x N_pairs]
        narrowband (bool): A flag to use the narrowband normalization instead of the broadband normalization.
        use_tqdm (bool): A flag to use the tqdm progress bar. Defaults to True.
        max_chunk (int): The maximum number of simultaneous matrix calculations. 
            Works best between 100-1000 but depends on the computer. Defaults to 300.

    Returns:
        np.ndarray: The pulsar pair covariance matrix for each frequency [N_freq x N_pairs x N_pairs] 
    """

    nfreq = len(Z[0])//2
    # The frequency selector matrices (set of diagonals)
    phitilde = np.repeat(np.diag(np.ones(nfreq)),2,axis=1)

    Ck = []
    iterable = tqdm(range(nfreq),desc='Freq covariances') if use_tqdm else range(nfreq)
    for k in iterable:
        # Same process as MCOS but per frequency instead
        mcos,_ = linear_solve(design_mat, np.diag(sig_abk[k]**2), rho_abk[k], None, 
                              method='diagonal')
        mcos = mcos[:,0]
        mcos[mcos<0] = 0

        norm_pow = mcos/np.sqrt(np.dot(mcos,mcos))
        est_pow = phi[2*k]*norm_pow

        cor_pow = np.sum([o*a for o,a in zip(orfs,est_pow)], axis=0)

        if narrowband:
            C = create_OS_pair_covariance(Z, phitilde[k], phitilde[k], cor_pow, norm_abk[k], False, max_chunk)
        else:
            # Since the "amplitude" S(f_k) is already passed in through ORF, we need
            # to remove the S(f_k) from phi
            C = create_OS_pair_covariance(Z, phitilde[k], phi/phi[2*k], cor_pow, norm_abk[k], False, max_chunk)
        Ck.append(C)

    return np.array(Ck)


#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------


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