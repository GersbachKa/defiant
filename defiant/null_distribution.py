
import numpy as np
import scipy.linalg as sl
from scipy.special import ndtr
from tqdm import tqdm

from . import utils

##------------------------------------------------------------------------------
## Null distribution functions
##------------------------------------------------------------------------------

def phase_shift_OS(os_obj, params=None, gamma=None, n_shifts=1000, use_tqdm=True):
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
    pars, idx, gam = os_obj._parse_params(params, 1, gamma)
    # Only one iteration is allowed, so lets unpack these
    pars, gam = pars[0], gam[0]
    phihat = utils.powerlaw(os_obj.freqs, 0, gam, 2)

    # Calculate the OS matrix products. Will need to shift X but not Z!
    X,Z = os_obj.compute_XZ(pars)

    # Compute the OS on this as the signal hypothesis
    rho_ab, sig_ab = os_obj.compute_rho_sig(X, Z, phihat)
    C = np.diag(sig_ab**2)

    A2,A2s = utils.linear_solve(os_obj.orf_design_matrix, C, rho_ab, 
                                None, method='diagonal')
    # SNR may be multi-component
    if np.array([A2]).size > 1:
        Cinv = np.diag(1/sig_ab**2) 
        fisher = os_obj.orf_design_matrix.T @ Cinv @ os_obj.orf_design_matrix
        target_snr = np.sqrt(A2.T @ fisher @ A2).item()
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the phase shifts
    null_snr = np.zeros(n_shifts)
    iterable = range(n_shifts) if not use_tqdm else tqdm(range(n_shifts), desc='shifts')
    for i in iterable:
        # Rotate X and Z
        X_rot, Z_rot = randomly_rotate_OS_matrices(X, Z)

        # Compute the OS with this rotation
        rho_rot, sig_rot = os_obj.compute_rho_sig(X_rot, Z_rot, phihat)
        C_rot = np.diag(sig_rot**2)

        A2_rot, A2s_rot = utils.linear_solve(os_obj.orf_design_matrix, C_rot, 
                                             rho_rot, None, method='diagonal')
        
        # SNR may be multi-component
        if np.array([A2]).size > 1:
            Cinv_rot = np.diag(1/sig_rot**2)
            fisher_rot = os_obj.orf_design_matrix.T @ Cinv_rot @ os_obj.orf_design_matrix
            null_snr[i] = np.sqrt(A2_rot.T @ fisher_rot @ A2_rot).item()
        else:
            null_snr[i] = np.squeeze(A2_rot/np.sqrt(A2s_rot)).item()

    # Now we have a measured SNR and a null distribution. We can compute the p-value
    pval = np.sum(null_snr >= target_snr)/n_shifts

    return pval, target_snr, null_snr


def sky_scramble_OS(os_obj, params=None, gamma=None, n_scrambles=1000, swap_pos=False, 
                    use_tqdm=True):
    """A function to compute the p-value of the OS using sky scrambles.

    This function compute the p-value, target SNR, and null distribution of the
    (non pair covariant) optimal statistic SNR using sky scrambling. This is done
    by creating a new, random design matrix and computing the OS using this.

    There are two methods to compute this:
    - If 'swap_pos' is False: The function will randomly create new pulsar positions
        and compute the design matrix using these positions. This creates a different
        distribution of pulsar separations, potentially underestimating the p-value.
    - If 'swap_pos' is True: The function will randomly swap pulsar positions and
        compute the design matrix using these positions. This maintains the same
        pulsar pair distribution, but causes the space of possible pulsar position
        configuration to be drastically lower. (i.e. with small numbers of pulsars
        the chance that 2 random configurations are similar is higher)

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
        n_scrambles (int): The number of sky scrambles to use. Defaults to 1000.
        swap_pos (bool): Whether to enable position swapping rather than creating
                new positions. Defaults to False.
        use_tqdm (bool): Whether to use tqdm for progress bar. Defaults to True.

    Returns:
        tuple: A tuple containing 3 elements:
            - pval (float): The p-value of the OS
            - target_snr (float): The measured SNR of the OS
            - null_snr (np.ndarray): The null distribution of the SNR
    """
    
    # First, get parameter dictionary, and gamma
    pars, idx, gam = os_obj._parse_params(params, 1, gamma)
    # Only one iteration is allowed, so lets unpack these
    pars, gam = pars[0], gam[0]
    phihat = utils.powerlaw(os_obj.freqs, 0, gam, 2)

    # Calculate the OS matrix products. Will need to shift X but not Z!
    X,Z = os_obj.compute_XZ(pars)

    # Compute the OS on this as the signal hypothesis
    rho_ab, sig_ab = os_obj.compute_rho_sig(X, Z, phihat)
    C = np.diag(sig_ab**2)

    A2,A2s = utils.linear_solve(os_obj.orf_design_matrix, C, rho_ab, 
                                None, method='diagonal')
    # SNR may be multi-component
    if np.array([A2]).size > 1:
        Cinv = np.diag(1/sig_ab**2) 
        fisher = os_obj.orf_design_matrix.T @ Cinv @ os_obj.orf_design_matrix
        target_snr = np.sqrt(A2.T @ fisher @ A2).item()
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the sky scrambles
    null_snr = np.zeros(n_scrambles)
    iterable = range(n_scrambles) if not use_tqdm else tqdm(range(n_scrambles), desc='scrambles')
    for i in iterable:
        # With scrambles, we can re-use the data and covariance, we just switch the
        # design matrix
        if swap_pos:
            f_xi, new_design = randomly_swap_pulsar_positions(os_obj)
        else:
            f_xi, new_design = randomly_create_pulsar_positions(os_obj)

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


def super_scramble_OS(os_obj, params=None, gamma=None, n_scrambles=1000, swap_pos=False, 
                    use_tqdm=True):
    """A function to compute the p-value of the OS using both phase shifts and sky scrambles.

    This function compute the p-value, target SNR, and null distribution of the
    (non pair covariant) optimal statistic SNR using both phase shifts and sky scrambling.
    This is done by creating a new, random design matrix and randomly phase shifting
    the rank-reduced fourier matrices X and Z (see phase_shift_OS() for details).

    This function also uses the 'swap_pos' argument to determine whether to use
    completely new pulsar positions or to swap pulsar positions (see sky_scramble_OS()).

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
        n_scrambles (int): The number of super scrambles to use. Defaults to 1000.
        swap_pos (bool): Whether to enable position swapping rather than creating
                new positions. Defaults to False.
        use_tqdm (bool): Whether to use tqdm for progress bar. Defaults to True.

    Returns:
        tuple: A tuple containing 3 elements:
            - pval (float): The p-value of the OS
            - target_snr (float): The measured SNR of the OS
            - null_snr (np.ndarray): The null distribution of the SNR
    """
    
    # First, get parameter dictionary, and gamma
    pars, idx, gam = os_obj._parse_params(params, 1, gamma)
    # Only one iteration is allowed, so lets unpack these
    pars, gam = pars[0], gam[0]
    phihat = utils.powerlaw(os_obj.freqs, 0, gam, 2)

    # Calculate the OS matrix products. Will need to shift X but not Z!
    X,Z = os_obj.compute_XZ(pars)

    # Compute the OS on this as the signal hypothesis
    rho_ab, sig_ab = os_obj.compute_rho_sig(X, Z, phihat)
    C = np.diag(sig_ab**2)

    A2,A2s = utils.linear_solve(os_obj.orf_design_matrix, C, rho_ab, 
                                None, method='diagonal')
    # SNR may be multi-component
    if np.array([A2]).size > 1:
        Cinv = np.diag(1/sig_ab**2) 
        fisher = os_obj.orf_design_matrix.T @ Cinv @ os_obj.orf_design_matrix
        target_snr = np.sqrt(A2.T @ fisher @ A2).item()
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    # Now we can start the super scrambles
    null_snr = np.zeros(n_scrambles)
    iterable = range(n_scrambles) if not use_tqdm else tqdm(range(n_scrambles), desc='s scrambles')
    for i in iterable:
        # Create a new design matrix
        if swap_pos:
            f_xi, new_design = randomly_swap_pulsar_positions(os_obj)
        else:
            f_xi, new_design = randomly_create_pulsar_positions(os_obj)

        # Create phase shifted data
        X_rot, Z_rot = randomly_rotate_OS_matrices(X, Z)

        # Compute the OS with this rotation
        rho_rot, sig_rot = os_obj.compute_rho_sig(X_rot, Z_rot, phihat)
        C_rot = np.diag(sig_rot**2)

        A2_scram, A2s_scram = utils.linear_solve(new_design, C_rot, rho_rot, None, 
                                                 method='diagonal')
        
        # SNR may be multi-component
        if np.array([A2]).size > 1:
            Cinv_rot = np.diag(1/sig_rot**2)
            fisher_scram = new_design.T @ Cinv_rot @ new_design
            null_snr[i] = np.sqrt(A2_scram.T @ fisher_scram @ A2_scram).item()
        else:
            null_snr[i] = np.squeeze(A2_scram/np.sqrt(A2s_scram)).item()

    # Now we have a measured SNR and a null distribution. We can compute the p-value
    pval = np.sum(null_snr >= target_snr)/n_scrambles

    return pval, target_snr, null_snr


def GX2(os_obj, params=None, gamma=None, max_iter=50, atol=1e-12, rtol=1e-12, pole_eps=1e-12):
    """Calculates the upper-tail probability P(rho_H0 >= rho) with the generalized-chi-squared (GX2) null distribution 
    (no-signal hypothesis) of the optimal statistic (OS) SNR using the Lugannani-Rice saddlepoint approximation.

    Constructs Q matrix of ORF-weighted pulsar cross-correlations from `build_Q_matrix`, then calculates its eigenvalues.
    These eigenvalues make up the null distribution (GX2) of the OS SNR. This distribution is used to 
    construct the cumulant generating function (CGF) and its derivatives. 

        K(t) = -0.5 * sum_i log(1 - 2*t*eigs[i])
    
    These are used to solve the saddlepoint equation K'(t) = rho, finding a saddlepoint t = t_hat 
    by safeguarded Newton-Raphson (bisection fallback). The saddlepoint and eigenvalues of the null distribution
    are used to calculate the Lugannani-Rice approximation to the upper-tail probability P(rho_H0 >= rho)

        p = 1 - Gaussian_CDF + (Gaussian_PDF * (1/u - 1/w))

    With u and w defined in terms of the saddlepoint and cumulants of the null distribution, 
    both encoding information about the GX2 distribution of the OS SNR. 

    Also samples from the null distribution of the SNR for plotting using `null_dist`, its doc comments are as follows:
        Draws SNRs from the null distirbution of SNRs

        Args:
            n_samples (int, optional):
                Number of samples to draw from null SNR distribution, defaults to 1.

        Returns:
            null_snrs (shape: n_samples)
                Drawn SNRs from null distribution.
    
    Notes: eigenvalues include both signs. Handles SNR above and below the null mean. Can take multiple rho values at once,
    all corresponding to the same PTA configuration; this is ideal for many realizations of one configuration. 

    Args:
        os_obj (defiant.core.OptimalStatistic): 
            The OptimalStatistic object.

        params (dict or int, optional): 
            The parameters or indexes to use. Check the documentation for usage info. 
            Defaults to None.
                
        gamma (float, optional): 
            The spectral index to use for the analysis. Check documentation for usage info. 
            Defaults to None.

        max_iter (int, optional):
            Maximum number of iterations for the Newton-Raphson solver. Default is 50.

        atol (float, optional):
            Absolute tolerance for convergence of the Newton-Raphson solver. Default is 1e-12.

        rtol (float, optional):
            Relative tolerance for convergence of the Newton-Raphson solver. Default is 1e-12.

        pole_eps (float, optional):
            Relative offset from eigenvalue bounds for the safeguarded Newton-Raphson root-solver. Default is 1e-12.

    Returns:
        p (array, shape [N,]): 
            P(rho_H0 >= rho); returns NaN where the solver did not converge
        
        target_SNR (float): 
            sinal-present SNR calculated from the os_obj

        eigs (array, shape [n_psrs*2*n_freq,]):
            eigenvalues of the Q matrix, which define the generalized chi-squared null distribution of the OS SNR

        null_dist (function):
            samples from the distribution of SNRs under the null hypothesis
    """

    # First, get parameter dictionary, and gamma
    pars, idx, gam = os_obj._parse_params(params, 1, gamma)
    # Only one iteration is allowed, so lets unpack these
    pars, gam = pars[0], gam[0]
    phihat = utils.powerlaw(os_obj.freqs, 0, gam, 2)

    # Calculate the OS matrix products. Will need to shift X but not Z!
    X,Z = os_obj.compute_XZ(pars)

    # Compute the OS on this as the signal hypothesis
    rho_ab, sig_ab = os_obj.compute_rho_sig(X, Z, phihat)
    C = np.diag(sig_ab**2)

    A2,A2s = utils.linear_solve(os_obj.orf_design_matrix, C, rho_ab,
                                None, method='diagonal')
    # Enforce single-component OS
    if np.array([A2]).size > 1:
        raise ValueError("Multi-component OS is not supported for the GX2.")
        # TODO: Explore implementations for this
    else:
        target_snr = np.squeeze(A2/np.sqrt(A2s)).item()

    Q0 = build_Q_matrix(
        os_obj.pair_idx,
        os_obj.orf_design_matrix[0],
        Z,
        phihat
    )
    
    eigs = np.linalg.eigvalsh(Q0)

    # Null cumulants
    mu0 = eigs.sum() # mean//1st cumulant evaluated at 0
    var0 = 2.0 * np.sum(eigs**2) # variance//2nd cumulant evaluated at 0
    sigma0 = np.sqrt(var0) # uncertainty
    kappa3 = 8.0 * np.sum(eigs**3) # 3rd cumulant evaluated at 0

    # Two Possible Brackets: 
    # rho > mu0 -> (0, t_pos)
    # OR 
    # rho < mu0 -> (t_neg, 0)
    t_neg = 0.5 / eigs.min() * (1.0 - pole_eps)
    t_pos = 0.5 / eigs.max() * (1.0 - pole_eps)
    above = target_snr > mu0
    lo = np.where(above, 0.0, t_neg)
    hi = np.where(above, t_pos, 0.0)

    # Gaussian initial guess, K'(t) ~= mu0 + var0*t 
    # else bracket midpoint (NaN fails the comparisons)
    t = (target_snr - mu0) / var0
    # finds if and where t is not above low & below hi, then replace with bisection midpoint
    t = np.where((t > lo) & (t < hi), t, 0.5 * (lo + hi))

    # if observed SNR == mu0 (null mean), then root is converged
    converged = target_snr == mu0

    # enforce saddlepoint and cumulants appropriately for these cases
    if converged:
        t = 0.0
        K = 0.0
        K2 = var0
    else:
        # tolerance for root finding, relative to SNR
        tol = atol + rtol * np.abs(target_snr)

        for it in range(max_iter + 1):

            # calculate first derivative of the CGF (K')
            d = 1.0 - 2.0 * t * eigs           # denom of K': shape = [1, eigs]
            a = eigs / d                       # matrix of all summation terms in K': shape = [1, eigs]
            f = a.sum() - target_snr     # sum over a's eigs dimension (columns) // f = K'(t) - SNR: shape = [1,]

            # If root has converged: record K and K''
            if np.abs(f) <= tol: # if residual is within relative tolerance, calculate cumulants
                K = -0.5 * np.log(d).sum() # CGF
                K2 = 2.0 * np.sum(a**2) # second derivative of CGF
                break
            else: # NaN are placeholders for an unconverged root
                K = np.nan 
                K2 = np.nan

            # If loop has reached max iterations, break loop
            if it == max_iter: 
                break

            # If unconverged: tighten bracket and propose Newton-Raphson proposal
            # Tighten bracket (K' increasing), then Newton step or bisection
            lo_i = np.where(f < 0.0, t, lo) # replace old lo with t if f < 0, true root must be between t and hi
            hi_i = np.where(f > 0.0, t, hi) # replace old hi with t if f > 0, true root must be between lo and t
            lo, hi = lo_i, hi_i # new bounds

            # propose Newton-Raphson proposal for all unconverged roots
            # NR-proposal: t_new = t - f(t)/f'(t) = t - f(t)/K''(t)
            # K''(t_i) = 2 * sum(a^2); sums over k for a_(ik) where i'th t
            t_nr = t - f / (2.0 * np.sum(a**2))
            t = np.where((t_nr > lo_i) & (t_nr < hi_i), t_nr, 0.5 * (lo_i + hi_i))

    # Lugannani-Rice approximation to Gaussian p-values
    # Note: (NaN K/K2 for an unconverged root propagates to NaN p)
    # suppress warnings for now, will account for errors by removing singularity at the mean
    with np.errstate(divide="ignore", invalid="ignore"):
        # calculate Lugannani-Rice variables

        # w = sqrt(2*(t*SNR - K(t)))
        # the inside of sqrt should be guranteed positive since K'(t) = SNR, 
        # but avoids floating point precision error with maximum()
        w = np.sign(t) * np.sqrt(2.0 * np.maximum(t * target_snr - K, 0.0))

        # u = t*sqrt(K''(t))
        u = t * np.sqrt(K2)

        # p = 1 - Gaussian_CDF + (Gaussian_PDF * (1/u - 1/w))
        # Uses `scipy.special.ndtr` in ndtr(w) to calculate CDF of a standard normal distribution
        # to the left of w, and since ndtr(-w) = 1 - ndtr(w) this calculates a simple Gaussian PDF 
        # more efficiently than using `scipy.norm.pdf()`
        p = ndtr(-w) + np.exp(-0.5 * w**2) / np.sqrt(2.0 * np.pi) * (1.0 / u - 1.0 / w)

    # Analytic limit of Lugannani-Rice near the null mean, removable singularity
    p_mean = 0.5 - kappa3 / (6.0 * sigma0**3 * np.sqrt(2.0 * np.pi))
    if np.abs(target_snr - mu0) < 1e-4 * sigma0:
        p = p_mean
        t = 0
    
    # Samples from the null distribution
    def null_dist(n_samples=1): 
        """Draws SNRs from the null distirbution of SNRs

        Args:
            n_samples (int, optional):
                Number of samples to draw from null SNR distribution, defaults to 1.

        Returns:
            null_snrs (shape: n_samples)
                Drawn SNRs from null distribution.
        """
        u = np.random.normal(0, 1, size=(n_samples, len(eigs))) # shape: [n_samples, n_eigs]
        null_snrs = np.sum(eigs[None,:]*(u**2), axis=1)
        return null_snrs


    return p, target_snr, eigs, null_dist, it
    



##------------------------------------------------------------------------------
## Helper functions
##------------------------------------------------------------------------------

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


def randomly_create_pulsar_positions(os_obj):
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
    f_psrs = [fake_psr() for i in range(len(os_obj.psrs))]

    # Get pulsar separations
    f_xi, _ = utils.compute_pulsar_pair_separations(f_psrs, os_obj.pair_idx)

    new_design_matrix = np.zeros_like(os_obj.orf_design_matrix.T)
    for i in range(os_obj.norfs):
        cur_orf = os_obj.orf_functions[i]
        try:
            # First see if we can supply the pulsar separations directly
            new_design_matrix[i] = cur_orf(f_xi)
        except:
            # Otherwise we need to supply the pulsar objects
            for j,(a,b) in enumerate(os_obj.pair_idx):
                v = cur_orf(f_psrs[a], f_psrs[b])
                new_design_matrix[i,j] = v
    
    return f_xi, new_design_matrix.T


def randomly_swap_pulsar_positions(os_obj):
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
    pos_list = np.array([p.pos for p in os_obj.psrs])

    # Randomly shuffle the positions
    np.random.shuffle(pos_list)

    # Create a list of fake pulsars
    f_psrs = [swap_psr(pos_list[i]) for i in range(len(os_obj.psrs))]

    # Get pulsar separations
    f_xi, _ = utils.compute_pulsar_pair_separations(f_psrs, os_obj.pair_idx)

    new_design_matrix = np.zeros_like(os_obj.orf_design_matrix.T)
    for i in range(os_obj.norfs):
        cur_orf = os_obj.orf_functions[i]
        try:
            # First see if we can supply the pulsar separations directly
            new_design_matrix[i] = cur_orf(f_xi)
        except:
            # Otherwise we need to supply the pulsar objects
            for j,(a,b) in enumerate(os_obj.pair_idx):
                v = cur_orf(f_psrs[a], f_psrs[b])
                new_design_matrix[i,j] = v
    
    return f_xi, new_design_matrix.T


def build_Q_matrix(pair_idx, Gamma_ab, Z, phihat): 
    """Constructs the normalized quadratic-form matrix Q for the
    single-component optimal statistic (OS).

    The matrix Q describes the ORF-weighted cross-correlations between pulsars in Fourier space. 
    Its eigenvalues determine the generalized chi-square null distribution of the OS SNR.

    For each pulsar, a square-root factor L_a of the Fourier-space covariance matrix Z_a
    is constructed using eigendecomposition. Negative eigenvalues are corrected by shifting 
    the spectrum so that the smallest eigenvalue is zero.

    For each pulsar pair (a, b), off-diagonal blocks are constructed as

        B_ab = Gamma_ab * L_a.T @ diag(phihat) @ L_b

    These blocks are assembled into a symmetric matrix Q with zero
    diagonal pulsar blocks. 

    Args:
        pair_idx (ndarray of int, shape [n_pairs, 2]): Pulsar index pairs 
        from the OS. Each row contains the indices (a, b) of one unique pulsar pair.

    Gamma_ab (ndarray of float, shape [n_pairs,]):
        Overlap reduction function evaluated for each
        pulsar pair in pair_idx. Ordering must match pair_idx.

    Z (ndarray of float, shape [n_psrs, 2*n_freq, 2*n_freq]):
        Fourier-space covariance matrices for each pulsar. Each Z_a is assumed symmetric 
        and is corrected to be positive semidefinite before constructing its square-root factor L_a.

    phihat (ndarray of float, shape [2*n_freq,]):
        Diagonal elements of the Fourier-space GWB cross-covariance
        template. Used to construct the pairwise matrices B_ab.

    Returns:
        Q (ndarray of float, shape [n_psrs*2*n_freq, n_psrs*2*n_freq]):
        Symmetric, unit-null-variance quadratic-form matrix containing the ORF-weighted 
        pulsar cross-correlations. Its eigenvalues define the generalized chi-square null SNR distribution.
    """
    # enforce single component OS for now

    # sqrt decomposition of Za (Batched eigenvalue decomp)
    evals, evecs = np.linalg.eigh(Z) # [npsr, 2nfreq] [npsr, 2nfreq, 2nfreq]

    # Clip negative eigenvalues (adding jitter)
    to_add = np.where(evals[:, 0] > 0, 0, -evals[:, 0]) # [npsr]
    evals += to_add[:, None]

    # Calculate a square root of all Z
    Ls = evecs * np.sqrt(evals)[:, None] # [npsr, 2nfreq, 2nfreq]

    n_psrs = len(Z)
    n_modes = Z.shape[1]

    # Make Btil_ab for each pair
    a, b = pair_idx.T # [npairs], [npairs]

    # Careful, we want L_a*Phi so use phihat[None, None, :]
    Btil_ab = (
        Ls[a].mT
        * phihat[None, None, :]
        ) @ Ls[b]

    # Multiply with the orf (this is where MCOS probably goes)
    B_ab = Gamma_ab[:, None, None] * Btil_ab # [npairs, 2nfreq, 2nfreq]

    # # Q is shape (n_psrs*n_modes X n_psrs*n_modes)
    Q = np.zeros((n_psrs*n_modes, n_psrs*n_modes))
    
    # Fill Q one pulsar block-row at a time
    for i in range(n_psrs):

        # Select pairs where pulsar i is the first pulsar
        mask = (a == i)

        # Column indices for the corresponding pulsars
        js = b[mask]

        # Row block of n_modes for pulsar i
        rows = (
            i * n_modes
            + np.arange(n_modes)
        )

        # Column block of n_modes for pulsar j
        cols = (
            js[:, None] * n_modes
            + np.arange(n_modes)[None, :]
        )

        # Fill upper-triangular combined block for i and j 
        # with appropriate B_ab entries
        Q[
            rows[None, :, None],
            cols[:, None, :]
        ] = B_ab[mask]

        # Fill corresponding symmetric lower-triangular blocks
        Q[
            cols[:, :, None],
            rows[None, None, :]
        ] = B_ab[mask].transpose(0, 2, 1)


    # For normalizing to unit null variance
    raw_var = (
        2.0
        * np.sum(
            Q**2
        )
    )

    # Normalization
    Q /= np.sqrt(
        raw_var
    )

    # Enforce Q to be symmetric
    Q = 0.5 * (
        Q + Q.T
    )

    return Q
