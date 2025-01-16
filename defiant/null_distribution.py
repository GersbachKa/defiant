
import numpy as np
import scipy.linalg as sl
from tqdm.auto import tqdm

from . import utils


def phase_shift_OS(OS_obj, params=None, gamma=None, n_shifts=1000, use_tqdm=True):
    """A function to compute the p-value of the OS using phase shifts.

    This function compute the p-value, target SNR, and null distribution of the
    (non pair covariant) optimal statistic SNR using phase shifts. This function
    uses the rank-reduced fourier representation of the optimal statistic to
    apply the rotation matrix formalism: X_rot = R @ X, Z_rot = R.T @ Z @ R

    This function also supports the multi-component optimal statistic through the
    multi-component total SNR calculated as sqrt(theta^T @ fisher @ theta), where
    theta is the measured multi-component amplitudes and fisher is the fisher
    information matrix.

    The 'params' argument behaves identically to the 'params' argument in the
    OS_obj.compute_OS() function. These usages are: 
        - If 'params' is left as None, then the function will use the maximum 
          likelihood parameters. Note that you can still set a fixed gamma value.
        - If 'params' is a dictionary, then the function will use these parameters.
          Note that you can still set a fixed gamma value.
        - If 'params' is a integer, then the function will interpret it as a chain
          index and use the parameters at that index. Note that you can still set
          a fixed gamma value.

    Do note that this function does NOT support noise marginalization. If you wish
    to perform noise marginalization, you should use this function for each iteration
    of the noise marginalization process.

    Args:
        OS_obj (defiant.core.OptimalStatistic): The OptimalStatistic object.
        params (dict or int, optional): The parameters or indexes to use. Check 
                the documentation for usage info. Defaults to None.
        gamma (float, optional): The spectral index to use for the analysis. Check
                documentation for usage info. Defaults to None.
        n_shifts (int): The number of phase shifts to use. Defaults to 1000.
        use_tqdm (bool): Whether to use tqdm for progress bar. Defaults to True.

    Returns:
        tuple: A tuple containing 3 elements:
            - pval (float): The p-value of the OS
            - target_snr (float): The measured SNR of the OS
            - null_snr (np.ndarray): The null distribution of the SNR
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
        Cinv = np.diag(1/sig_ab**2) 
        fisher = OS_obj.orf_design_matrix.T @ Cinv @ OS_obj.orf_design_matrix
        target_snr = np.sqrt(A2.T @ fisher @ A2).item()
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the phase shifts
    null_snr = np.zeros(n_shifts)
    iterable = range(n_shifts) if not tqdm else tqdm(range(n_shifts), desc='shifts')
    for i in iterable:
        # Rotate X and Z
        X_rot, Z_rot = randomly_rotate_OS_matrices(X, Z)

        # Compute the OS with this rotation
        rho_rot, sig_rot = OS_obj._compute_rho_sig(X_rot, Z_rot, phihat)
        C_rot = np.diag(sig_rot**2)

        A2_rot, A2s_rot = utils.linear_solve(OS_obj.orf_design_matrix, C_rot, 
                                                 rho_rot, None, method='diagonal')
        
        # SNR may be multi-component
        if np.array([A2]).size > 1:
            Cinv_rot = np.diag(1/sig_rot**2)
            fisher_rot = OS_obj.orf_design_matrix.T @ Cinv_rot @ OS_obj.orf_design_matrix
            null_snr[i] = np.sqrt(A2_rot.T @ fisher_rot @ A2_rot).item()
        else:
            null_snr[i] = np.squeeze(A2_rot/np.sqrt(A2s_rot)).item()

    # Now we have a measured SNR and a null distribution. We can compute the p-value
    pval = np.sum(null_snr >= target_snr)/n_shifts

    return pval, target_snr, null_snr


def sky_scramble_OS(OS_obj, params=None, gamma=None, n_scrambles=1000, swap_pos=False, 
                    use_tqdm=True):
    
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
        Cinv = np.diag(1/sig_ab**2) 
        fisher = OS_obj.orf_design_matrix.T @ Cinv @ OS_obj.orf_design_matrix
        target_snr = np.sqrt(A2.T @ fisher @ A2).item()
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the phase shifts
    null_snr = np.zeros(n_scrambles)
    iterable = range(n_scrambles) if not tqdm else tqdm(range(n_scrambles), desc='shifts')
    for i in iterable:
        # With scrambles, we can re-use the data and covariance, we just switch the
        # design matrix
        if swap_pos:
            f_xi, new_design = randomly_swap_pulsar_positions(OS_obj)
        else:
            f_xi, new_design = randomly_create_pulsar_positions(OS_obj)

        A2_scram, A2s_scram = utils.linear_solve(new_design, C, rho_ab, None, 
                                             method='diagonal')
        
        # SNR may be multi-component
        if np.array([A2]).size > 1:
            fisher_scram = new_design.T @ Cinv @ new_design
            null_snr[i] = np.sqrt(A2_scram.T @ fisher_scram @ A2_scram).item()
        else:
            null_snr[i] = np.squeeze(A2_scram/np.sqrt(A2s_scram)).item()

    # Now we have a measured SNR and a null distribution. We can compute the p-value
    pval = np.sum(null_snr >= target_snr)/n_scrambles

    return pval, target_snr, null_snr


def super_scramble():
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


def randomly_create_pulsar_positions(OS_obj):
    """A function to randomly create a new design matrix using random pulsar positions

    This function first randomly generates new pulsar positions then computes the
    pulsar separations and the design matrix using the new positions.

    Note that this function only works for isotropic ORFs and is faster with pre-defined
    ORFs. This function does support the use of custom ORFs, but the only attributes
    that the pulsars will have will be the .pos attribute.

    Args:
        OS_obj (defiant.core.OptimalStatistic): The OptimalStatistic object that we want to
            create a new design matrix for.

    Returns:
        tuple: A tuple containing the new pulsar separations and the new design matrix.
            - f_xi (np.ndarray): The new pulsar separations.
            - new_design_matrix (np.ndarray): The new design matrix.
    """
    # Since I intend use utils.compute_pulsar_pair_separations(), we need
    # to create a dummy pulsar object with the .pos attribute (cartesian coordinates)
    class fake_psr:
        def __init__(self):
            self.name = 'fake_psr'
            # We need to get random positions for the pulsars, 
            # Use normal distribution sampling in 3d then normalize the vectors.
            # Need to be careful around (0,0,0), exclude points within 1e-6 of the origin
            self.pos = None
            while self.pos is None:
                pos = np.random.normal(0,1,3)

                if np.sum(pos**2) > 1e-3:
                    self.pos = pos / np.sqrt(np.sum(pos**2))
    
    # Create a list of fake pulsars
    f_psrs = [fake_psr() for i in range(len(OS_obj.psrs))]

    # Get pulsar separations
    f_xi, _ = utils.compute_pulsar_pair_separations(f_psrs, OS_obj._pair_idx)

    new_design_matrix = np.zeros_like(OS_obj.orf_design_matrix.T)
    for i in range(OS_obj.norfs):
        cur_orf = OS_obj.orf_functions[i]
        try:
            # First see if we can supply the pulsar separations directly
            new_design_matrix[i] = cur_orf(f_xi)
        except:
            # Otherwise we need to supply the pulsar objects
            for j,(a,b) in enumerate(OS_obj._pair_idx):
                v = cur_orf(f_psrs[a], f_psrs[b])
                new_design_matrix[i,j] = v
    
    return f_xi, new_design_matrix.T


def randomly_swap_pulsar_positions(OS_obj):
    """A function to randomly make a new design matrix using SWAPPED pulsar positions

    This function first randomly swaps pulsar positions then computes the
    pulsar separations and the design matrix using the new positions. This is useful
    for testing if the pulsar positions are important to the detection statistic.

    Note that this function only works for isotropic ORFs and is faster with pre-defined
    ORFs. This function does support the use of custom ORFs, but the only attributes
    that the pulsars will have will be the .pos attribute.

    Args:
        OS_obj (defiant.core.OptimalStatistic): The OptimalStatistic object that we want to
            create a new design matrix for.

    Returns:
        tuple: A tuple containing the new pulsar separations and the new design matrix.
            - f_xi (np.ndarray): The new pulsar separations.
            - new_design_matrix (np.ndarray): The new design matrix.
    """
    # Since I intend use utils.compute_pulsar_pair_separations(), we need
    # to create a dummy pulsar object with the .pos attribute (cartesian coordinates)
    class swap_psr:
        def __init__(self,pos):
            self.name = 'fake_psr'
            self.pos = pos

    # Get a list of positions
    pos_list = np.array([p.pos for p in OS_obj.psrs])

    # Randomly shuffle the positions
    np.random.shuffle(pos_list)

    # Create a list of fake pulsars
    f_psrs = [swap_psr(pos_list[i]) for i in range(len(OS_obj.psrs))]

    # Get pulsar separations
    f_xi, _ = utils.compute_pulsar_pair_separations(f_psrs, OS_obj._pair_idx)

    new_design_matrix = np.zeros_like(OS_obj.orf_design_matrix.T)
    for i in range(OS_obj.norfs):
        cur_orf = OS_obj.orf_functions[i]
        try:
            # First see if we can supply the pulsar separations directly
            new_design_matrix[i] = cur_orf(f_xi)
        except:
            # Otherwise we need to supply the pulsar objects
            for j,(a,b) in enumerate(OS_obj._pair_idx):
                v = cur_orf(f_psrs[a], f_psrs[b])
                new_design_matrix[i,j] = v
    
    return f_xi, new_design_matrix.T