
import numpy as np
import scipy.linalg as sl

from .custom_exceptions import *


def linear_solve(X,C,r,method=None):
    """A simple method for minimizing the chi-square 
    
    This function minimizes (r - X*theta)^T C^(-1) (r - X*theta), where r is the 
    data column vector (N x 1) , X is a design matrix (N x M), and C is the 
    covariance matrix (N x N). This function will analytically minimize to find
    the solution vector (M x 1).
    
    method must be one of the following:
        - 'diagonal': Supplied C is the diagonal covariance matrix
        - 'exact': Use exact matrix inverses
        - 'pinv': Use pseudo matrix inverses

    Args:
        X (ndarray): The design matrix (N x M)
        C (ndarray): The covariance matrix (N x N) or diagonal matrix (N x 1)
        r (ndarray): The residual vector (N x 1)
        method (str, optional): The method of solving to use

    Raises:
        NameError: If the method name cannot be identified.

    Returns:
        theta (ndarray): The solution vector theta (M x 1)
        cov (ndarray): The covariance matrix between linear elements (M x M)
    """
    # Valid methods: exact, pinv, cholesky, Regularized SVD
    # TODO: Implement other methods

    if method is None:
        # TODO: Automate method switching
        if len(C.shape)>1:
            # Assume square matrix
            condition_num = np.linalg.cond(C)
            method = 'pinv'
        else:
            # Assume the diagonal covariance
            method = 'diagonal'

    if method.lower() in ['exact','pinv','diagonal']:
        
        if method.lower() == 'diagonal':
            fisher = X.T @ np.diag(1/C[:]) @ X
            dirty_map = X.T @ np.diag(1/C[:]) @ r

        elif method.lower() == 'exact':
            Cinv = np.linalg.inv(C)
            fisher = (X.T @ Cinv @ X)
            dirty_map = (X.T @ Cinv @ r)

        else:
            Cinv = np.linalg.pinv(C)
            fisher = (X.T @ Cinv @ X)
            dirty_map = (X.T @ Cinv @ r)

        if fisher.size>1:
            cov = np.linalg.pinv(fisher)
        else:
            cov = 1/fisher
        
        theta = cov.T @ dirty_map
        return theta, cov

    else:
        msg = f'Unknown method \'{method}\' for linear solving.'
        raise NameError(msg)
    

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
        gwb_sig = [s for s in pta._signalcollections[0] if s.signal_id==gwb_name][0]
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
        gwb_signal = [s for s in pta._signalcollections[0] if s.signal_id==gwb_name][0]
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
        dict: A dictionary of the maximum likelihood parameters
    """
    burn = lfcore.burn
    lnlike = lfcore.chain[burn:,lfcore.params.index('lnlike')]
    max_i = burn + np.argmax(lnlike)
    values = lfcore.chain[max_i]
    params = {p:v for p,v in zip(lfcore.params,values)}

    return params
    

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

    gwb_signal = [s for s in pta._signalcollections[0] if s.signal_id==gwb_name][0]
    rho = gwb_signal.get_phi(pars)

    a2,_ = linear_solve(model_phi, cov, rho, None)
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


def binned_pair_correlations(xi, rho, sig, bins=10):
    """Create binned separation vs correlations with even pairs per bin.

    This function creates a binned version of the xi, rho, and sig values to better
    vizualize the correlations as a function of pulsar separation. This function uses
    even number of pulsar pairs per bin. Note that this function only works with continuous 
    ORFs in pulsar separation space.

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        rho (numpy.ndarray): A vector of pulsar pair correlated amplitude
        sig (numpy.ndarray): A vector of uncertainties in rho
        bins (int): Number of bins to use. Defaults to 10.

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


def binned_pair_covariant_correlations(xi, rho, Sigma, bins=10, orf='hd'):
    """Create binned separation vs correlations with pulsar pair covariances.

    This function creates a binned version of the xi, rho, and sig values to better
    vizualize the correlations as a function of pulsar separation while including
    some of the effects of pulsar pair covariances. Note that this does not account 
    for correlations between the different bins. This function uses even number 
    of pulsar pairs per bin. Note that this function only works with continuous 
    ORFs in pulsar separation space.
    Also note that orf can be replaced with a custom function which must accept 
    pulsar positions (cartesian) as its only 2 arguments.
    Predefined orf names are:
        'hd' - Hellings and downs
        'dipole' - Dipole
        'monopole' - Monopole
        'gw_dipole' - Gravitational wave dipole
        'gw_monopole' - Gravitational wave monopole
        'st' - Scalar transverse

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        rho (numpy.ndarray): A vector of pulsar pair correlated amplitude
        Sigma (numpy.ndarray): The pulsar pair covariance matrix
        orf (str, function): The name of a predefined ORF function or custom function 
        bins (int): Number of bins to use. Defaults to 10.

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
    
    for i in range(bins):
        #Mask and select range of values to average
        l,h = int(ranges[i]), int(ranges[i+1])
        subXi = (xi[sortMask])[l:h]
        subRho = (rho[sortMask])[l:h]
        subSig = (Sigma[sortMask,:][:,sortMask])[l:h,l:h]
        subORF = orf_xi(subXi,orf)[:,None]

        r,s2 = linear_solve(subORF,subSig,subRho,'pinv')

        xiavg[i] = np.average(subXi)
        bin_orf = orf_xi(xiavg[i],orf)
        rhoavg[i] = bin_orf * r
        sigavg[i] = np.abs(bin_orf)*np.sqrt(s2)
    
    return xiavg, rhoavg, sigavg


def orf_xi(xi, orf='hd'):
    """A function to turn pulsar separations into correlations using a set ORF

    Given a pulsar separation or separations, compute the correlation factor
    for that separation and given overlap reduction function. Note that orf can be 
    replaced with a custom function which must accept pulsar positions (cartesian) 
    as its only 2 arguments.
    Predefined orf names are:
        'hd' - Hellings and downs
        'dipole' - Dipole
        'monopole' - Monopole
        'gw_dipole' - Gravitational wave dipole
        'gw_monopole' - Gravitational wave monopole
        'st' - Scalar transverse

    Args:
        xi (numpy.ndarray or float): A vector or float of pulsar pair separation(s)
        orf (str, function): The name of a predefined ORF function or custom function 

    Raises:
        ValueError: If given a string of an unrecognized ORF

    Returns:
        _type_: correlation(s) for the pair separation(s)
    """
    if type(orf) == str:
        from enterprise_extensions import model_orfs
        if orf.lower() == 'hd':
            orf_func = model_orfs.hd_orf
        elif orf.lower() == 'dipole':
            orf_func = model_orfs.dipole_orf
        elif orf.lower() == 'monopole':
            orf_func = model_orfs.monopole_orf
        elif orf.lower() == 'gw_dipole':
            orf_func = model_orfs.gw_dipole_orf
        elif orf.lower() == 'gw_monopole':
            orf_func = model_orfs.gw_monopole_orf
        elif orf.lower() == 'st':
            orf_func = model_orfs.st_orf
        else:
            raise ValueError(f"Undefined ORF name '{orf}'")
        orf_func
    else:
        orf_func = orf
    
    orf_lamb = lambda x: orf_func([1,0,0], [np.cos(x),np.sin(x),0])

    if np.array(xi).size>1:
        return np.array([orf_lamb(x) for x in xi])
    else:
        return orf_lamb(xi)    
