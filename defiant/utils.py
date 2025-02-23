
import numpy as np
import scipy.linalg as sl
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from enterprise.constants import fyr as f_year

from .custom_exceptions import *
from .orf_functions import get_orf_function, get_pulsar_separation


def linear_solve(X, C, r, s=None,  method=None, fisher_diag=False):
    """A method for minimizing the chi-square for a gaussian likelihood.
    
    This function minimizes (r - X*theta)^T C^(-1) (r - X*theta), where r is the 
    data column vector (N x 1) , X is a design matrix (N x M), and C is the 
    covariance matrix (N x N). This function will analytically minimize to find
    the solution vector (M x 1). 
    This function also allows users to only use the diagonal elements of the fisher 
    matrix by setting fisher_diag to True. This is useful for cases where you 
    wish to compute many single model fits with a single calculation.
    
    method must be one of the following:
        - 'diagonal': The covariance matrix is diagonal
        - 'exact': Use exact matrix inverses (np.linalg.inv)
        - 'pinv': Use pseudo matrix inverses (np.linalg.pinv)
        - 'cholesky': Use cholesky decomposition for forward substitution (scipy.linalg.cho_factor)
        - 'solve': Use numpy.linalg.solve for forward substitution (np.linalg.solve)
        - 'woodbury': Use the Woodbury matrix identity to invert C^-1 = (A + K)^-1 
            where A is a diagonal matrix and K is a dense matrix. This requires the
            's' parameter to be the diagonal components of A.

    Args:
        X (ndarray): The design matrix (N x M)
        C (ndarray): The covariance matrix (N x N)
        r (ndarray): The residual vector (N x 1)
        s (ndarray): The diagonal components of the A matrix for the 
            Woodbury method (N x N). Defaults to None.
        method (str, optional): The method of solving to use. Defaults to diagonal 
            if C is diagonal or pinv if dense.
        fisher_diag_only (bool): Whether to zero the off-diagonal elements of the
            Fisher matrix. Defaults to False.

    Raises:
        NameError: If the method name cannot be identified.
        ValueError: If the method is 'woodbury' and the shape of C is not [2 x N_pairs x N_pairs]

    Returns:
        theta (ndarray): The solution vector theta (M x 1)
        cov (ndarray): The covariance matrix between linear elements (M x M)
    """
    X = X[:,None] if len(X.shape) == 1 else X
    r = r[:,None] if len(r.shape) == 1 else r

    if method is None: # Basic automatic method selection
        method = 'diagonal' if is_diagonal(C) else 'pinv'
        
    if method.lower() in ['exact','pinv','diagonal','solve','cholesky','woodbury']:
        
        if method.lower() == 'diagonal':
            Cinv = np.diag(1/np.diag(C))
            fisher = X.T @ Cinv @ X
            dirty_map = X.T @ Cinv @ r

        elif method.lower() == 'exact':
            Cinv = np.linalg.inv(C)
            fisher = X.T @ Cinv @ X
            dirty_map = X.T @ Cinv @ r

        elif method.lower() == 'pinv':
            Cinv = np.linalg.pinv(C)
            fisher = X.T @ Cinv @ X
            dirty_map = X.T @ Cinv @ r

        elif method.lower() == 'solve':
            fisher = X.T @ np.linalg.solve(C,X)
            dirty_map = X.T @ np.linalg.solve(C,r)

        elif method.lower() == 'cholesky':
            cf = cho_factor(C)
            fisher = X.T @ cho_solve(cf,X)
            dirty_map = X.T @ cho_solve(cf,r)

        elif method.lower() == 'woodbury':
            # This method requires 's' = A, the diagonally dominant element of C
            # For the PTA optimal statistic case, this is just np.diag(sig_ab**2)
            if s is None:
                raise ValueError('Woodbury method requires the diagonal components of A to be supplied!')
            
            # Use same notation as https://en.wikipedia.org/wiki/Woodbury_matrix_identity
            A = s
            K = C - A
            In = np.eye(A.shape[0])
        
            cinv = woodbury_inverse(A,In,In,K)

            fisher = X.T @ cinv @ X
            dirty_map = X.T @ cinv @ r 

        else:
            msg = f'Unknown method \'{method}\' for linear solving.'
            raise NameError(msg)
            

        if fisher.size>1:
            if fisher_diag:
                cov = np.diag( 1/np.diag(fisher) )
            else:
                cov = np.linalg.pinv(fisher)
        else:
            cov = 1/fisher
        
        theta = cov.T @ dirty_map
        return theta, cov

    else:
        msg = f'Unknown method \'{method}\' for linear solving.'
        raise NameError(msg)
    

def powerlaw(fgw, log10_A=0.0, gamma=13./3., modes=1):
    """A function to create a power-law Power Spectral Density (PSD).

    Given the non-repeating set of gravitational wave frequencies, this function
    will calculate the power-law PSD using the given log amplitude and spectral index.
    Setting modes>1 will add duplicate frequencies to the PSD. (i.e. if you have sine and
    cosine modes for each frequency, set modes=2).

    This uses the equation: S(f) = A**2 / (12 * pi**2 * f**3) * (f/f_year)**(-gamma) * df
    where A is the amplitude, gamma is the spectral index, f is the frequency, and df 
    is the frequency bin width.


    Args:
        fgw (np.ndarray): The (non-repeating) gravitational wave frequencies. [n]
        log10_A (float): The log_10 amplitude of the power-law. Defaults to 0.0.
        gamma (float): The spectral index. Defaults to 13./3..
        modes (int): The number of modes to use. Defaults to 1.

    Returns:
        np.ndarray: The power-law PSD at the given frequencies. [modes*n]
    """

    # Calculate frequency bin widths. Bin 0 starts at f=0 and is up to f[0]
    df = np.diff(fgw,prepend=[0])

    # Calculate the powerlaw (PSD)
    pl = 10**(2*log10_A) * (1/(12 * np.pi**2 * f_year**3)) * (fgw/f_year)**(-gamma) * df

    if modes>1:
        return np.repeat(pl,modes)
    else:
        return pl
    

def get_pta_frequencies(pta, gwb_name='gw'):
    """Gets the basis frequencies for the GWB Fourier design matrix.

    Uses a solution adapted from a solution found in
    enterprise_extensions.frequentist.optimal_statistic._get_freqs().
    Note that this function only returns the sine frequencies (i.e ignoring
    the cosine component frequencies)

    Raises:
        ModelPTAError: If the function is unable to get the GWB frequencies

    Returns:
        np.array: The frequencies of the GWB
    """
    try:
        gwb_sig = [s for s in pta._signalcollections[0].signals if s.signal_id==gwb_name][0]
        gwb_sig.get_basis()
 
        if type(gwb_sig._labels) == dict:
            freqs = gwb_sig._labels['']
        else: 
            freqs = gwb_sig._labels

        return freqs[::2]
    
    except Exception as e:
        raise ModelPTAError('Unable to get PTA frequencies!') from e
        

def get_fixed_gwb_gamma(pta, gwb_name='gw'):
    """A simple function to grab the GWB gamma value from a fixed gamma PTA model.

    This function grabs the power-law gravitational wave background spectral index
    from a fixed-gamma Enterprise PTA model.

    Args:
        pta (enterprise.signals.signal_base.PTA): The fixed-gamma PTA model
        gwb_name (str, optional): The GWB name given in the PTA model. Defaults to 'gw'.

    Raises:
        KeyError: If this method of grabing gamma fails

    Returns:
        float: The value of the GWB gamma 
    """
    try:
        gwb_signal = [s for s in pta._signalcollections[0].signals if s.signal_id==gwb_name][0]
        gamma = gwb_signal.get(gwb_name+'_gamma')
    except KeyError as e:
        msg = "Unable to get gamma value from the PTA model!"
        raise KeyError(msg) from e
          
    return gamma 


def get_max_like_params(lfcore):
    """A function to get the maximum likelihood parameters from a la_forge core.

    This function will return a maximum likelihood parameter dictionary from
    a la_forge.core.Core() object.

    Args:
        lfcore (la_forge.core.Core): A core object containing the MCMC chains

    Returns:
        tuple: A tuple of:
            - A dictionary of the maximum likelihood parameters
            - The index of the maximum likelihood in the la_forge.core.Core() 
            chain (including burn-in)
    """
    burn = lfcore.burn
    lnlike = lfcore.chain[burn:,lfcore.params.index('lnlike')]
    max_i = burn + np.argmax(lnlike)
    values = lfcore.chain[max_i]
    params = {p:v for p,v in zip(lfcore.params,values)}

    return params, max_i
    

def compute_pulsar_pair_separations(psrs, pair_idx=None):
    """A function to compute the pairwise pulsar separations in radians.

    This simple function computes all pairs of pulsar separations in radians. 
    If pair_idx is not None, this function will compute the separations of the 
    in the order of the list of pairs. If pair_idx is None, then this function
    also returns the names of the pulsar pairs.
    Note: pair_idx must be [N_pairs x 2] or None.

    Args:
        psrs (list): A list of enterprise.pulsar.BasePulsar objects
        pair_idx (numpy.ndarray, optional): An ordered list of pulsar pairs. 
                to calculate pair separations. Use none for a defaults order to None.

    Returns:
        numpy.ndarray: An array of pulsar pair separations
        list: A list of the pulsar pair names
    """
    
    if pair_idx is None:
        pair_idx = np.array([(a,b) for a in range(len(psrs)) for b in range(a+1,len(psrs))])
    
    pair_names = [(psrs[a].name, psrs[b].name) for (a,b) in pair_idx]

    xi = np.arccos([np.dot(psrs[a].pos, psrs[b].pos) for (a,b) in pair_idx])

    return xi, pair_names


def fit_a2_from_params(params, pta=None, gwb_name='gw', model_phi=None, cov=None):
    """A function to get an estimated GWB amplitude at 1/yr given parameters.

    A function which can estimate the GWB amplitude at 1/yr given any model of a pta.
    Supplying gamma and freqs is optional and saves function calls if you already
    have these. cov is the estimated covariance between different frequencies. Set
    this to None to default to equal weighting.

    Args:
        params (dict): A dictionary of parameter name:value pairs
        pta (enterprise.signals.signal_base.PTA): The PTA model
        gwb_name (str, optional): The name of the GWB in the PTA model. Defaults to 'gw'.
        gamma (float, optional): The spectral index for the power-law model fit.
        freqs (numpy.ndarray, optional): The set of frequencies for which the PTA is modeling.
        cov (numpy.ndarray, optional): The estimated covariance between frequencies.
                defaults to the Identity matrix.

    Returns:
        float: The estimated square amplitude of the GWB.
    """
    # Check if the amplitude is in the params!
    if gwb_name+'_log10_A' in params:
        return 10**(2*params[gwb_name+'_log10_A'])
    
    if model_phi is None:
        from enterprise.signals.utils import powerlaw
        freqs = get_pta_frequencies(pta, gwb_name)
        gamma = get_fixed_gwb_gamma(pta, gwb_name)
        model_phi = powerlaw(freqs, 0, gamma, 1)

    if cov is None:
        cov = np.diag(np.ones(len(model_phi)))

    pars = freespec_param_fix(params, gwb_name)

    gwb_signal = [s for s in pta._signalcollections[0].signals if s.signal_id==gwb_name][0]
    rho = gwb_signal.get_phi(pars)

    a2,_ = linear_solve(model_phi, cov, rho)
    return a2.item()


def get_a2_estimate(params, freqs, gamma, gwb_name, lfcore=None):
    """A function to get the estimated GWB amplitude at 1/yr given parameters.

    This function is intended for use for the pair covariance amplitude estimation
    for the GWB. This function will return the estimated amplitude at 1/yr given
    the parameters, frequencies, and gamma. 
    There are many ways this function can be used:
        - If the amplitude is in the params and gamma is not, this function will
        return the this amplitude squared. Note that this assumes that the gamma
        used in the model is the same as 'gamma'.
        - If the amplitude and gamma are in the params, this function will return
        the amplitude squared. Even if the gamma is different from the one used in
        the model, this function will recalculate the amplitude for the new gamma.
        - If the amplitude is not in the params, this function will assume that the
        model is a freespec model and will calculate the amplitude by fitting the
        a powerlaw model to the rho parameters. If lfcore is supplied, this function
        will weight the amplitude by the variance in the parameter estimates. Otherwise,
        this function will assume equal weighting.

    Args:
        params (dict): A dictionary of parameter name:value pairs
        freqs (np.ndarray): The set of frequencies for which the PTA is modeling.
        gamma (float): The spectral index for the power-law model amplitude fit.
        gwb_name (str): The name of the GWB in the PTA model such that 
            gwb_name+'_log10_A' is the amplitude.
        lfcore (la_forge.core.Core, optional): The core object made from a freespec MCMC.

    Returns:
        float: The estimated square amplitude of the GWB.
    """

    if gwb_name+'_log10_A' in params:
        # Amplitude is in the params dictionary!
        if gwb_name+'_gamma' not in params:
            # Gamma is not in the params, assume that 'gamma' was used to measure
            return 10**(2*params[gwb_name+'_log10_A'])

        # Gamma in params
        if params[gwb_name+'_gamma'] == gamma:
            # Gamma given is the same as the one in the params (no re-calculation needed)
            return 10**(2*params[gwb_name+'_log10_A'])
        
        # Gamma is in the params but is different. Need to recalculate
        model = powerlaw(freqs, 0, gamma, 1)
        rho = powerlaw(freqs, params[gwb_name+'_log10_A'], params[gwb_name+'_gamma'], 1)
        cov = np.diag(np.ones(len(freqs)))
        
    else:
        # Amplitude is not present, assume we have a freespec
        pars = freespec_param_fix(params, gwb_name) # Add the gwb_log10_rho key

        model = powerlaw(freqs, 0, gamma, 1)
        rho = 10**(2*pars[gwb_name+'_log10_rho'])
        if lfcore is not None:
            # Weight by the variance in the parameter estimates
            cov = freespec_covariance(lfcore, gwb_name)
        else:
            # Equal weighting
            cov = np.diag(np.ones(len(freqs)))
    
    a2,_ = linear_solve(model, cov, rho, method='pinv')
    return a2.item()


def freespec_param_fix(params, gwb_name):
    """A simple function to fix a slight issue with free spectral models in Enterprise.

    This function will create a new parameter dictionary with a new key value pair
    which corresponds to the GWB log10 rho parameters as Enterprise wants
    {'gw_log10_rho':numpy.ndarray} rather than the MCMC given {'gw_log10_rho_0':val,...}

    Args:
        params (dict): A parameter dictionary
        gwb_name (str): The name of the GWB in the parameter dictionary

    Returns:
        dict: An updated dictionary with the free-spec fix
    """
    rho_name = gwb_name+'_log10_rho_'
    nfreqs = int( sum(1 for k in params.keys() if rho_name in k) )

    if nfreqs>0:
        pars = {}
        pars.update(params)

        vals = [params[rho_name+f'{i}'] for i in range(nfreqs)]

        pars.update( {gwb_name+'_log10_rho':np.array(vals)} )
        return pars
    else:
        return params
    

def freespec_covariance(lfcore, gwb_name):
    """A basic function to get the covariance between frequencies from an MCMC

    This function takes a la_forge.core.Core object which *must* contain a freespec
    mcmc and a gwb_name and creates a covariance matrix between the frequencies.

    Args:
        lfcore (la_forge.core.Core): The core object made from a freespec MCMC.
        gwb_name (str): The name of the GWB in the parameter names. 

    Returns:
        numpy.ndarray: The estimated covariance matrix between frequencies.
    """
    nfreqs = sum(1 for f in lfcore.params if gwb_name+'_log10_rho_' in f)
    
    chain = [lfcore.get_param(gwb_name+f'_log10_rho_{i}') for i in range(nfreqs)]
    cov = np.cov(chain)

    return cov


def binned_pair_correlations(xi, rho, sig, bins=10, orf='hd'):
    """Create binned separation vs correlations with even pairs per bin.

    This function creates a binned version of the xi, rho, and sig values to better
    vizualize the correlations as a function of pulsar separation. This function uses
    even number of pulsar pairs per bin. Note that this function only works with continuous 
    ORFs in pulsar separation space. If given a sig which has a shape of [N_pairs x N_pairs],
    this function will assume you have supplied a pair covariance matrix and will use
    equation [35] from Gersbach et al. 2024. Also note that orf can be replaced with 
    a custom function which must accept pulsar positions (cartesian) as its only 2 arguments.
    The pre-defined ORFs found in defiant.orf_functions.defined_orfs are:
        - 'hd' or 'hellingsdowns': Hellings and Downs
        - 'dp' or 'dipole': Dipole
        - 'mp' or 'monopole': Monopole
        - 'gwdp' or 'gw_dipole': Gravitational wave dipole
        - 'gwmp' or 'gw_monopole': Gravitational wave monopole
        - 'st' or 'scalar_tensor': Scalar tensor

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        rho (numpy.ndarray): A vector of pulsar pair correlated amplitude
        sig (numpy.ndarray): A vector of uncertainties in rho OR a covariance matrix
        bins (int): Number of bins to use. Defaults to 10.
        orf (str, function): The name of a predefined ORF function or custom function
            orf is only used if sig is a covariance matrix

    Returns:
        xiavg (numpy.ndarray): The average pulsar separation in each bin
        rhoavg (numpy.ndarray): The weighted average pulsar pair correlated amplitudes
        sigavg (numpy.ndarray): The uncertainty in the weighted average pair amplitudes
    """
    temp = np.arange(0,len(xi),len(xi)/bins,dtype=np.int16)
    ranges = np.zeros(bins+1)
    ranges[0:bins]=temp
    ranges[bins]=len(xi)
    
    xiavg = np.zeros(bins)
    rhoavg = np.zeros(bins)
    sigavg = np.zeros(bins)
    
    #Need to sort by pulsar separation
    sortMask = np.argsort(xi)
    

    if len(sig.shape)>1:
        # Pair covariant
        orf_func = get_orf_function(orf)
        for i in range(bins):
            #Mask and select range of values to average
            l,h = int(ranges[i]), int(ranges[i+1])
            subXi = (xi[sortMask])[l:h]
            subRho = (rho[sortMask])[l:h]
            subSig = (sig[sortMask,:][:,sortMask])[l:h,l:h]
            subORF = orf_func(subXi)[:,None]

            r,s2 = linear_solve(subORF,subSig,subRho,method='pinv')

            xiavg[i] = np.average(subXi)
            bin_orf = orf_func(xiavg[i])
            rhoavg[i] = bin_orf * r
            sigavg[i] = np.abs(bin_orf)*np.sqrt(s2)

    else:
        # Non pair covariant
        for i in range(bins):
            #Mask and select range of values to average
            subXi = xi[sortMask]
            subXi = subXi[int(ranges[i]):int(ranges[i+1])]
            subRho = rho[sortMask]
            subRho = subRho[int(ranges[i]):int(ranges[i+1])]
            subSig = sig[sortMask]
            subSig = subSig[int(ranges[i]):int(ranges[i+1])]
        
            subSigSquare = np.square(subSig)
        
            xiavg[i] = np.average(subXi)
            rhoavg[i] = np.sum(subRho/subSigSquare)/np.sum(1/subSigSquare)
            sigavg[i] = 1/np.sqrt(np.sum(1/subSigSquare))
    
    return xiavg,rhoavg,sigavg  


def calculate_mean_sigma_for_MCOS(xi, A2, A2_cov, orfs=['hd','dipole','monopole'], 
                                  n_samples=1000, clip_thresh=1e-10):
    """Calculate the mean and sigma of the total MCOS fit for a given xi.

    For a given pulsar pair separation, xi (can be a vector of xi), this function
    will calculate the mean and sigma of the total correlated power for a given
    MCOS fit A2 vector and A2_cov covariance matrix. You will also need to supply
    the ORFs you want to use in the calculation. 
    The pre-defined ORFs found in defiant.orf_functions.defined_orfs are:
        - 'hd' or 'hellingsdowns': Hellings and Downs
        - 'dp' or 'dipole': Dipole
        - 'mp' or 'monopole': Monopole
        - 'gwdp' or 'gw_dipole': Gravitational wave dipole
        - 'gwmp' or 'gw_monopole': Gravitational wave monopole
        - 'st' or 'scalar_tensor': Scalar tensor

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        A2 (numpy.ndarray): A vector of A^2 values
        A2_cov (numpy.ndarray): A covariance matrix between A^2 values
        orfs (list, optional): A list of ORFs to use (either names or custom functions).
        n_samples (int): The number of samples of the fit to generate. Defaults to 1000.
        clip_thresh (float): The threshold to clip the covariance matrix if needed. Defaults to 1e-10.
    
    Returns:
        (np.ndarray,np.ndarray): The corresponding mean and 1-sigma standard deviation.
    """
    clipped = clip_covariance(A2_cov,clip_thresh)
    norm = multivariate_normal(mean=A2,cov=clipped,allow_singular=True)
    rvs = norm.rvs(size=n_samples)

    orf_mods = [get_orf_function(o)(xi) for o in orfs]

    mod_vals = []
    for A2 in rvs:
        mod_vals.append(np.sum([a*o for a,o in zip(A2,orf_mods)],axis=0))

    return np.mean(mod_vals,axis=0), np.std(mod_vals,axis=0)


def uncertainty_sample(A2,A2s,pfos=False,mcos=False,n_usamples=100):
    """A function to generate the full A^2 distribution from means and 1-sigma errors

    A function to implement uncertainty sampling to account for underlying 
    uncertainty in the optimal statistic A^2. This function uses Gaussians (or 
    multivariate Gaussians for MCOS) at each point. This function also works for
    PFOS, MCOS, any combination. 

    NOTE: The expected shape of A2 and A2s are dependent on pfos and mcos flags.
    *N is the number of noise marginalized iterations, M is the number of orf models,
    and F is the number of frequencies. (These are default output shapes of defiant!)

    If pfos is False and mcos is False: 
        A2 and A2s are [N] and [N] respectively (i.e. arrays)
    If pfos is False and mcos is True:
        A2 and A2s are [N x M] and [N x M] respectively
    If pfos is True and mcos is False:
        A2 and A2s are [N x F] and [N x F] respectively
    If pfos is True and mcos is True:
        A2 and A2s are [N x F x M] and [N x F x M] respectively
        

    Args:
        A2 (np.ndarray): An array of A2 or Sk values from the OS or PFOS respectively
        A2s (np.ndarray): An array of A2s or Sks values from the OS or PFOS respectively
        pfos (bool): A flag to use a PFOS version. Defaults to False.
        mcos (bool): A flag to use a MCOS version. Defaults to False.
        n_usamples (int, optional): The number of random samples for each NMOS iteration. 
            Defaults to 100.

    Returns:
        tot_A2: An array of the total A^2 or S(f_k) distribution
            The shape of the return is:
                [N_samples] if pfos is False and mcos is False
                [N_samples x M] if pfos is False and mcos is True
                [N_samples x F] if pfos is True and mcos is False
                [N_samples x F x M] if pfos is True and mcos is True
                
    """
    if pfos:
        if mcos:
            # PF+NM+MC+OS
            nm_iter = A2.shape[0]
            n_freq = A2.shape[1]
            n_orf = A2.shape[2]

            all_A2 = np.zeros((nm_iter*n_usamples,n_freq,n_orf))
            for i in range(nm_iter):
                for j in range(n_freq):
                    mv_norm = np.random.multivariate_normal(A2[i,j,:],A2s[i,j,:],size=(n_usamples))
                    all_A2[i*n_usamples:(i+1)*n_usamples,j,:] = mv_norm
            
        else:
            # PF+NM+OS
            nm_iter = A2.shape[0]
            n_freq = A2.shape[1]

            all_A2 = np.zeros((nm_iter*n_usamples,n_freq))
            for i in range(nm_iter):
                for j in range(n_freq):
                    mv_norm = np.random.normal(A2[i,j],A2s[i,j],size=(n_usamples))
                    all_A2[i*n_usamples:(i+1)*n_usamples,j] = mv_norm

    else: # OS
        if mcos:
            # NM+MC+OS
            nm_iter = A2.shape[0]
            n_orf = A2.shape[1]

            all_A2 = np.zeros((nm_iter*n_usamples,n_orf))
            for i in range(nm_iter):
                mv_norm = np.random.multivariate_normal(A2[i,:],A2s[i,:],size=(n_usamples))
                all_A2[i*n_usamples:(i+1)*n_usamples,:] = mv_norm
            
        else:
            # NM+OS
            nm_iter = A2.shape[0]

            all_A2 = np.zeros((nm_iter*n_usamples))
            for i in range(nm_iter):
                norm = np.random.normal(A2[i],A2s[i],size=(n_usamples))
                all_A2[i*n_usamples:(i+1)*n_usamples] = norm
    
    return all_A2

            
def clip_covariance(cov, eig_thresh=1e-10):
    """A function to clip the small or negative eigenvalues of a covariance matrix

    This function is to fix some minor numerical problems with some covariance matrices
    by clipping negative and very small eigenvalues of a correlation matrix made from
    the given covariance matrix and setting them to the specified threshold. If the 
    covariance matrix is already postive definite, this function will return the
    original matrix. 

    Args:
        cov (np.ndarray): The covariance matrix to fix
        eig_thresh (float): The threshold value of the correlation matrix eigenvalues 
            to clip as a ratio of the largest. Defaults to 1e-10.

    Returns:
        np.ndarray: The clipped covariance matrix
    """
    e_vals,e_vec = np.linalg.eigh(cov)
    e_thr = eig_thresh*np.max(e_vals)
    if np.any(e_vals<e_thr):
        e_vals[e_vals<e_thr] = e_thr
        new_cov = e_vec@np.diag(e_vals)@np.linalg.pinv(e_vec)

        if np.any(np.linalg.eigvals(new_cov)<0):
            raise ValueError('Covariance matrix is still not positive definite! Clipping is too small!')

        return new_cov
    
    return cov


def chi_square(rho, sig, design_mat, a2):
    """A function to calculate the chi-square value of a model fit

    Args:
        rho (np.ndarray): The data vector [N]
        sig (np.ndarray): The uncertainty or covariance in the data vector [N] or [N x N]
        design_mat (np.ndarray): The design matrix [N x M]
        a2 (np.ndarray): The model vector [M]

    Returns:
        float: The floating point chi-square value
    """
    if sig.ndim == 1:
        sig = np.diag(1/sig**2)
    if a2.ndim == 1:
        a2 = a2[:,None]
    if design_mat.ndim == 1:
        design_mat = design_mat[:,None]
    if rho.ndim == 1:
        rho = rho[:,None]
    
    model = design_mat @ a2
    residuals = (rho - model)

    chi2 = residuals.T @ sig @ residuals
    return chi2.item()


def invert_skymap(hp_map):
    """A function to change between GW propogation direction and GW source direction.

    This function takes a healpy map or maps which has either GW propogation direction 
    or GW source direction and swaps to the other.  This function supports lists or 
    arrays of healpy maps indicated to the function with len(hp_map.shape)>1.

    Function from MAPS.utils.invert_omega() by Nihan Pol.

    Args:
        hp_map (np.ndarray or list): A healpy map or array or list of healpy maps

    Returns:
        list: A list of healpy maps or a single healpy map with inverted direction
    """
    import healpy as hp

    arg_arr = np.array(hp_map)
    if len(arg_arr.shape) == 1:
        arg_arr = arg_arr[None, :]

    inv_map = []
    for mp in arg_arr:
        nside = hp.get_nside(mp)

        all_pix_idx = np.array([ ii for ii in range(hp.nside2npix(nside = nside)) ])

        all_pix_x, all_pix_y, all_pix_z = hp.pix2vec(nside = nside, ipix = all_pix_idx)

        inv_all_pix_x = -1 * all_pix_x
        inv_all_pix_y = -1 * all_pix_y
        inv_all_pix_z = -1 * all_pix_z

        inv_pix_idx = [hp.vec2pix(nside = nside, x = xx, y = yy, z = zz) for (xx, yy, zz) in zip(inv_all_pix_x, inv_all_pix_y, inv_all_pix_z)]
        inv_map.append(mp[inv_pix_idx])

    if len(inv_map) == 1:
        return inv_map[0]
    
    return inv_map


def woodbury_inverse(A, U, C, V, ret_cond = False):
    """A function to compute the inverse of a matrix using the Woodbury matrix identity.

    This function computes the inverse of a matrix using the Woodbury matrix identity 
    for more stable matrix inverses. The matrix labels are the same as those used 
    in its wikipedia page (https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
    This solves the inverse to: (A + UCV)^-1.
    
    This function can also solve inverses of the form: (A + K)^-1 where A is a
    diagonal matrix and K is a dense matrix. To do this, simply set U and C to 
    the identity matrix and V to K.
    
    Args:
        A (np.ndarray): An invertible nxn matrix
        U (np.ndarray): A nxk matrix
        C (np.ndarray): An invertible kxk matrix
        V (np.ndarray): A kxn matrix
        ret_cond (bool, optional): Whether to return the condition number of the matrix (C + V A^-1 U)

    Returns:
        np.ndarray: The inverse of the matrix (A + UCV)^-1
    """

    Ainv = np.diag( 1/np.diag(A) )
    Cinv = np.linalg.pinv(C)

    # (A+UCV)^-1 = (A^-1) - A^-1 @ U @ (C^-1 + V @ A^-1 @ U)^-1 @ V @ A^-1
    CVAU = Cinv + V @ Ainv @ U
    tot_inv = Ainv - Ainv @ U @ np.linalg.solve(CVAU, V @ Ainv) 

    if ret_cond:
        return tot_inv, np.linalg.cond(CVAU)
    
    return tot_inv

def is_diagonal(C):
    """A function to check if a matrix is diagonal.

    Args:
        C (np.ndarray): A 2D matrix to check for diagonality

    Returns:
        bool: True if the matrix is diagonal, False otherwise
    
    Raises:
        ValueError: If the matrix is not 2D
    """
    if len(C.shape) != 2:
        raise ValueError('Matrix must be 2D!')
    return np.all( C == np.diag(np.diag(C)) )