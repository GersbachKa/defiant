
from .os_Exceptions import *

import numpy as np
import scipy.linalg as sl

from enterprise.signals.utils import powerlaw

from warnings import warn


def linear_solve(X,C,r,method=None):
    # Valid methods: exact, pinv, cholesky, SVD, Regularized SVD
    condition_num = np.linalg.cond(C)

    if method is None:
        # TODO: Automate method switching
        method = 'pinv'

    if method.lower() in ['exact','pinv']:
        # Not super necessary, as numpy will use exact with pinv() if a 
        # matrix has an analytic inverse.
        Cinv = np.linalg.inv(C) if method.lower()=='exact' else np.linalg.pinv(C)
        
        fisher = (X.T @ Cinv @ X)
        dirty_params = (X.T @ Cinv @ r)

        covariance = np.linalg.inv(fisher) if method.lower()=='exact' else np.linalg.pinv(fisher)
        theta_hat = covariance.T @ dirty_params
            
        # I need a citation for this...
        total_snr = np.sqrt(theta_hat.T @ fisher @ theta_hat)
        return theta_hat, covariance, total_snr.item()

    elif method.lower() == 'cholesky':
        pass
    elif method.lower() == ' SVD':
        pass
    else:
        msg = f'Unknown method \'{method}\' for linear solving.'
        raise NameError(msg)


def check_pta_params(pta, params, gwb_nfreq, gwb_name='gw'):
    # Do a quick check and ensure each parameter in the PTA is in params
    for par in pta.param_names:
        if par not in params.keys():
            msg = f'Parameter {par} not in parameter dictionary! This may '+\
                   'cause problems with getting certain matrices' 
            warn(msg)  
        
    # Check for a free-spec
    if gwb_name+'_log10_rho_0' in params.keys():
        pars = {}
        pars.update(params)

        pname = gwb_name+'_log10_rho'
        vals = [params[pname+f'_{i}'] for i in range(gwb_nfreq)]

        pars.update({pname:np.array(vals)})
        return pars
    else:
        return params
    

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
    psr_sig = pta[0]
    gwb_sig = psr_sig[gwb_name]
    gwb_sig.get_basis()

    if type(gwb_sig._labels) == dict:
        freqs = gwb_sig._labels['']
    elif type(gwb_sig._labels) == np.ndarray:
        freqs = gwb_sig._labels
    else:
        raise ModelPTAError('Unable to get PTA frequencies! Check to make sure gwb_name is set!')
        
    return freqs[::2]


def get_gwb_gamma(pta, params, gwb_name='gw'):

    if gwb_name+'_gamma' in params:
        gamma = params[gwb_name+'_gamma']
    else:
        # Assume the PTA is fixed gamma. If so, use that.
        try:
            gamma = pta[0][gwb_name].get(gwb_name+'_gamma')
        except KeyError:
            msg = "Unknown gamma value from the PTA model!"
            raise KeyError(msg)        
    return gamma 


def get_gwb_a2(pta, params, gwb_name='gw', gamma=13./3., lfcore=None):
    f = get_pta_frequencies(pta, gwb_name)
    pars = check_pta_params(pta, params, len(f), gwb_name)
    return _gwb_a2_from_freqs(pars, gamma, f, gwb_name, lfcore)


def _gwb_a2_from_freqs(params, gamma, frequencies, gwb_name='gw', lfcore=None):
    # Check which version of a CURN process we have.
    # Should either be power-law, or free-spectral.

    if gwb_name+'_log10_A' in params:
        # Amplitude is in the parameter dictionary! Easy peasy!
        return 10**(2*params[gwb_name+'_log10_A'])
        
    elif gwb_name+'_log10_rho_0' in params or gwb_name+'_log10_rho':
        # Free-spectral model. We need to do fitting instead.
        model = powerlaw(frequencies,0,gamma,1)[:,None]

        rho = params[gwb_name+f'_log10_rho']

        if lfcore is not None:
            # Get the covariance from the la forge core
            chain = [lfcore.get_param(gwb_name+f'_log10_rho_{i}') 
                            for i in range(len(frequencies))]
            cov = np.cov(chain)
        else:
            cov = np.diag(np.ones(len(frequencies)))

        a2,_,_ = linear_solve(model,cov,rho[:,None])
        return a2.item()
    else:
        msg = "Unable to get an estimated amplitude for the GWB"
        raise ModelPTAError(msg)
    

def get_max_likelihood_params(lfcore):
    if lfcore is not None:
        burn = lfcore.burn
        lnlike = lfcore.chain[burn:,lfcore.params.index('lnlike')]
        max_i = burn + np.argmax(lnlike)
        values = lfcore.chain[max_i]
        params = {p:v for p,v in zip(lfcore.params,values)}
        return params
    else:
        msg = 'No chains supplied! Cannot compute maximum likelihoods!'
        raise BadParametersError(msg)
    
def compute_pulsar_separations(psrs, pair_idx = None):
    if pair_idx is None:
        pair_idx = np.array([(a,b) for a in range(len(psrs)) for b in range(a+1,len(psrs))])

    xi = np.arccos([np.dot(psrs[i].pos,psrs[j].pos) for (i,j) in pair_idx])

    return xi