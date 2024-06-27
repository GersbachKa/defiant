
from .ospp_exceptions import *
from .utils import linear_solve

import numpy as np

from itertools import combinations, combinations_with_replacement
from tqdm import tqdm

def _compute_pair_covariance(Z, phi1, phi2, orf, norm_ab, a2_est, use_tqdm, max_chunk):
    fact_c = _factored_pair_covariance(Z,phi1,phi2,orf,norm_ab,use_tqdm,max_chunk)
    return fact_c[0] + a2_est*fact_c[1] + a2_est**2*fact_c[2]


def _compute_mcos_pair_covariance(Z, phi1, phi2, orf, design, rho_ab, sig_ab, 
                                  norm_ab, a2_est, use_tqdm, max_chunk):
    # MCOS w/ pair covariance - From Sardesai et al. 2023
    # Need to compute the a no-PC MCOS for amp estimates
    temp_c = np.diag(np.square(sig_ab))
    mcos,_,_ = linear_solve(design,temp_c,rho_ab,'pinv')
    est_pow = a2_est*(mcos/np.sum(mcos))

    # Hijack the factored code by giving it correlated power in ORF and A2=1!
    cor_pow = np.sum([o*a for o,a in zip(orf,est_pow)], axis=1)

    fact_c = _factored_pair_covariance(Z,phi1,phi2,cor_pow,norm_ab,use_tqdm,max_chunk)
    return fact_c[0] + fact_c[1] + fact_c[2]


# Hidden function which directly calculates the amplitude-factored pair covariance matrix.
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
        phi1 (numpy.ndarray): A 2N_frequencies array of the estimator's spectral model
        phi2 (numpy.ndarray): A 2N_frequencies array of the estimated spectral shape
        orf (numpy.ndarray): A N_pulsar x N_pulsar matrix of the ORF (or correlated power) 
                for each pair of pulsars
        norm_ab (numpy.ndarray): The N_pair array of normalizations of the pair-wise estimators 
        use_tqdm (bool, optional): Whether to use TQDM's progress bar. Defaults to False.
        max_chunk (int, optional): The maximum number of simultaneous matrix calculations. 
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


    def case1(a,b,c,d): #(ab,cd)
        a0 = np.zeros_like(a)
        a2 = np.zeros_like(a)
        a4 = orf[a,c]*orf[d,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[c,d]) + \
             orf[a,d]*orf[c,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[d,c])
        return [a0,a2,a4]
    
    def case2(a,b,c): #(ab,ac)
        a0 = np.zeros_like(a)
        a2 = orf[b,c]*np.einsum('ijk,ikj->i', Zphi1[b],Zphi1Zphi2[a,c])
        a4 = orf[a,c]*orf[a,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[c,a])
        return [a0,a2,a4]

    def case3(a,b): #(ab,ab)
        a0 = np.einsum('ijk,ikj->i',Zphi1[b],Zphi1[a])
        a2 = np.zeros_like(a)
        a4 = orf[a,b]**2 * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[b,a])
        return [a0,a2,a4]

    
    # --------------------------------------------------------------------------
    # Now lets calculate them!
    C_m = np.zeros((3,len(pairs_idx),len(pairs_idx)),dtype=np.float64)

    if use_tqdm:
        from tqdm import tqdm
        ntot = len(PoP)
        progress = tqdm(total=ntot,desc='Pairs of pairs',ncols=80)


    try: # This is used to close the progress bar if an exception occurs

        # Case1: no matching pulsars--------------------------------------------
        mask = (~psr_match[:,0] & ~psr_match[:,1]) & \
               (~psr_inv_match[:,0] & ~psr_inv_match[:,1])
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,c,d = PoP[mask].T
    
        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case1(a[l:h],b[l:h],c[l:h],d[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))
    

        # Case2: 1 matching pulsar----------------------------------------------
        mask = (psr_match[:,0] & ~psr_match[:,1]) # Check for (ab,ac)
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,c = PoP[mask].T

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))


        mask = (~psr_inv_match[:,0] & psr_inv_match[:,1]) # Check for (ab,bc)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,_,c = PoP[mask].T # Index swap a with b

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))


        mask = (~psr_match[:,0] & psr_match[:,1]) # Check for (ab,cb)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,c,_ = PoP[mask].T # Index swap a with b

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))

        # Case3: 2 matching pulsars---------------------------------------------
        mask = psr_match[:,0] & psr_match[:,1] # Check for (ab,ab)
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,_ = PoP[mask].T 

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case3(a[l:h],b[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h])) 

    except Exception as e:
        if use_tqdm: progress.close()
        msg = 'Exception occured during pair covariance creation!'
        raise PCOSInteruptError(msg) from e
        
    if use_tqdm: progress.close()

    # Include the final sigmas
    C_m[:] *= np.outer(norm_ab**2,norm_ab**2)

    return C_m
