
from .os_Exceptions import *

import numpy as np

from itertools import combinations_with_replacement, combinations
from tqdm import tqdm


def _compute_mcos_pair_covariance(Z, phi1, phi2, orf_mat, sig_ab, pair_idx, 
                                  mcos_pow_est, use_tqdm=True, max_chunk=300):
    # Hijack the factored calculation for MCOS
    corpow = np.sum([pow*orfm for pow,orfm in zip(mcos_pow_est,orf_mat)],axis=0)
    fact_C = _compute_factored_pair_covariance(Z, phi1, phi2, corpow, sig_ab, 
                                               pair_idx, use_tqdm, max_chunk)
    return np.sum(fact_C,axis=0)


def _compute_factored_pair_covariance(Z, phi1, phi2, orf_mat, sig_ab, pair_idx, use_tqdm=True, max_chunk=300):
    # Use some pre-calculations to speed up processing
    a,b = pair_idx[:,0], pair_idx[:,1]

    npsr = len(Z)
    nfreq = len(Z[0])

    pairs_idx = np.array(list( combinations(range(npsr),2) ),dtype=int)

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
        a4 = orf_mat[a,c]*orf_mat[d,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[c,d]) + \
             orf_mat[a,d]*orf_mat[c,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[d,c])
        return [a0,a2,a4]
    
    def case2(a,b,c): #(ab,ac)
        a0 = np.zeros_like(a)
        a2 = orf_mat[b,c]*np.einsum('ijk,ikj->i', Zphi1[b],Zphi1Zphi2[a,c])
        a4 = orf_mat[a,c]*orf_mat[a,b] * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[c,a])
        return [a0,a2,a4]

    def case3(a,b): #(ab,ab)
        a0 = np.einsum('ijk,ikj->i',Zphi1[b],Zphi1[a])
        a2 = np.zeros_like(a)
        a4 = orf_mat[a,b]**2 * np.einsum('ijk,ikj->i', Zphi1Zphi2[b,a], Zphi1Zphi2[b,a])
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
        raise InteruptedPairCovariance(msg) from e
        
    if use_tqdm: progress.close()

    # Include the final sigmas
    C_m[:] *= np.outer(sig_ab**2,sig_ab**2)

    return C_m
