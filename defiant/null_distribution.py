
import numpy as np
import scipy.linalg as sl
from tqdm.auto import tqdm

from . import utils




def phase_shift_OS(OS_obj, params=None, gamma=None, n_shifts=100, use_tqdm=True):
    """A method to compute the p-value of the OS using phase shifts.

    This method compute the p-value, target SNR, and null distribution of the
    (non pair covariant) optimal statistic SNR using phase shifts. This method
    uses the rank-reduced fourier representation of the optimal statistic to
    apply the rotation matrix formalism: X_rot = R @ X, Z_rot = R.T @ Z @ R

    Do note that this function does NOT support noise marginalization, or
    multiple components

    Args:
        OS_obj (_type_): _description_
        params (_type_, optional): _description_. Defaults to None.
        gamma (_type_, optional): _description_. Defaults to None.
        n_shifts (int, optional): _description_. Defaults to 100.
        use_tqdm (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    # First, get parameter dictionary, and gamma
    pars, idx, gam = OS_obj._parse_params(params, 1, gamma)
    # Only one iteration is allowed, so lets unpack these
    pars, gam = pars[0], gam[0]
    phihat = utils.powerlaw(OS_obj.freqs, 0, gam, 2)

    # Calculate the OS matrix products. Will need to shift X but not Z!
    X,Z = OS_obj._compute_XZ(pars)

    # Compute the OS on this as the signal hypothesis
    rho_ab, sig_ab = OS_obj._compute_rho_sig(X, Z, phihat)
    C = np.diag(sig_ab**2)

    A2,A2s = utils.linear_solve(OS_obj.orf_design_matrix, C, rho_ab, 
                                None, method='diagonal')
    # SNR may be multi-component
    if np.array([A2]).size > 1:
        target_snr = np.sqrt(A2.T @ A2s @ A2).item()
        print('Here')
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the phase shifts
    null_snr = np.zeros(n_shifts)
    iterable = range(n_shifts) if not tqdm else tqdm(range(n_shifts), desc='shifts')
    for i in iterable:
        # Get the rotation matrix for each pulsar

        # Rotate X and Z
        X_rot, Z_rot = randomly_rotate_OS_matrices(X, Z)

        # Compute the OS with this rotation
        rho_rot, sig_rot = OS_obj._compute_rho_sig(X_rot, Z_rot, phihat)
        C_rot = np.diag(sig_rot**2)

        A2_rot, A2s_rot = utils.linear_solve(OS_obj.orf_design_matrix, C_rot, 
                                                 rho_rot, None, method='diagonal')
        
        # SNR may be multi-component
        if np.array([A2]).size > 1:
            null_snr[i] = np.sqrt(A2_rot.T @ A2s_rot @ A2_rot).item()
        else:
            null_snr[i] = np.squeeze(A2_rot/np.sqrt(A2s_rot)).item()

    # Now we have a measured SNR and a null distribution. We can compute the p-value
    pval = np.sum(null_snr >= target_snr)/n_shifts

    return pval, target_snr, null_snr


def super_scramble(self, n=100, params=None):
    pass


def GX2(self):
    raise NotImplementedError("GX2 not implemented yet.")
    





def randomly_rotate_OS_matrices(X, Z):
    """A function to randomly rotate (phase shift) the OS matrices X and Z

    This function will randomly rotate the OS matrices X and Z by a random
    phase shift. This is done by creating a block diagonal matrix of rotation
    matrices R, and applying them to each pulsar's X and Z such that:
    X_rot = R @ X, Z_rot = R.T @ Z @ R

    Args:
        X (np.ndarray): The X matrix of the OS [npsr, 2*nfreq]
        Z (np.ndarray): The Z matrix of the OS [npsr, 2*nfreq, 2*nfreq]

    Returns:
        tuple: A tuple of the rotated X and Z matrices:
            X_rot (np.ndarray): The rotated X matrix
            Z_rot (np.ndarray): The rotated Z matrix
    """
    
    # Get the number of pulsars and frequencies
    npsr, nfreq = X.shape[0], X.shape[1]//2 # Divide by 2 due to sine and cosine 

    X_rot = np.zeros(X.shape)
    Z_rot = np.zeros(Z.shape)
    for i in range(npsr):
        # Create a random rotation matrix using a random phi
        rand_phi = np.random.uniform(0, 2*np.pi, nfreq)

        cos_phi = np.cos(rand_phi)
        sin_phi = np.sin(rand_phi)

        # Create sub-blocks of block diagonal matrix
        blocks = np.zeros((nfreq, 2, 2))
        blocks[:,0,0] = cos_phi
        blocks[:,1,0] = -sin_phi
        blocks[:,0,1] = sin_phi
        blocks[:,1,1] = cos_phi

        # block_diag requires each block to be its own argument. We can unpack the 
        # blocks by using list comprehension and the * operator. 
        # (numpy doesn't support unpacking natively)
        R = sl.block_diag(*[b for b in blocks])

        # Rotate the matrices
        X_rot[i] = R @ X[i]
        Z_rot[i] = R.T @ Z[i] @ R
        
    return X_rot, Z_rot