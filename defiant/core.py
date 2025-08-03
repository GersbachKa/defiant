
from . import custom_exceptions as os_ex
from . import utils
from . import pair_covariance as pc 
from . import null_distribution as os_nu
from . import orf_functions as orf_funcs
from .caching_code import product_cache, set_cache_params

import numpy as np
import scipy.linalg as sl

import enterprise.signals as ent_sig
from enterprise.pulsar import BasePulsar

from la_forge.core import Core

from tqdm import tqdm
from warnings import warn


class OptimalStatistic:
    """A class to compute the various forms of the Optimal Statistic for a given PTA.

    This class is designed in such a way to be able to combine all of the various
    generalizations of the Optimal Statistic into a single, cohesive class. This class
    can be made to compute any of the froms shown in the Defiant choice tree in the
    documentation of Defiant.

    Once constructed, use `compute_OS()` and `compute_PFOS()`!

    Attributes:
        psrs (list): A list of `enterprise.pulsar.BasePulsar` objects.
        npsr (int): The number of pulsars in the PTA.
        pta (enterprise.signals.signal_base.PTA): A PTA object.
        gwb_name (str): The name of the GWB in the PTA object.
        marginalizing_timing_model (bool): Whether the PTA is using a marginalizing timing model.
        psr_signals (list): A list of the pulsar signals in the PTA. (pta._signalcollections)
        lfcore (la_forge.core.Core): A la_forge.core.Core object.
        max_like_params (dict): The maximum likelihood parameters from the chain.
        max_like_idx (int): The index of the maximum likelihood parameters in the chain.
        freqs (np.ndarray): The frequencies of the PTA. [Nfreq]
        nfreq (int): The number of frequencies in the PTA.
        pair_idx (np.ndarray): The index of the pulsar pairs. [Npair x 2]
        pair_names (list): A list of the used pulsar pairs. [Npair x 2]
        npair (int): The number of pulsar pairs.
        norfs (int): The number of overlap reduction functions.
        orf_functions (list): A list of the ORF functions.
        orf_design_matrix (np.ndarray): The design matrix of the ORFs. [Norfs x Npair]
        orf_names (list): The names of the ORFs. [Norfs]
        orf_matrix (np.ndarray): A pulsar matrix of ORF values. [Norfs x Npsr x Npsr]
        mcos_orf (np.ndarray): The ORF for the pair covariance matrix if using
            the experimental MCOS. [Npair x Npair]
        nmos_iterations (dict): A dictionary holding outputs from compute_OS() and compute_PFOS().
        sub_tqdm (bool): Whether to use progress bars on frequency and pair covariance calculations.
        max_matrix_chunk (int): The maximum number of simultaneous matrix products.
        _get_phi (function): A function to get the GWB phi from the PTA for given parameters.
        _cache (dict): A dictionary of cached matrix products
        _cache_{'func_name'} (list): A list of cached results for each pulsar.
    """

    def __init__(self, psrs, pta, gwb_name='gw', core_path=None, core=None,  
                 chain_path=None, chain=None, param_names=None, 
                 orfs=['hd'], orf_names=None, pcmc_orf=None, clip_z=None,
                 sub_tqdm=False, pre_cache=True):
        """Initializes the OptimalStatistic object.

        There are many ways to initialize the OptimalStatistic object, and most
        parameters are optional. The most basic way to initialize the object is to
        call this initializer with a list of pulsars `psrs`, a PTA object `pta`, and
        the name of the gwb `gwb_name`. You may also need to use the `set_chain_params()` 
        and `set_orf()` methods to set the MCMC chains and ORFs respectively. 
        For convienence, the parameters for these methods are also available here 
        in the initializer.

        For info on the `corepath`, `core`, `chain_path`, `chain`, and `param_names` 
        check documentation of `set_chain_params()`

        For info on the `orfs`, and `orf_names` check documentation of `set_orf()`.

        The `pre_cache` parameter is a boolean that determines whether to pre-cache
        the many matrix products. This was default behavior for previous versions of 
        Defiant and only moves the computational time to the initialization instead
        of the first function calls.

        NOTE: The OS assumes that the GW spectrum is identically modeled in all
        pulsars such that phi is the same in all pulsars.

        **Experimental**: The `clip_z` parameter is an experimental feature that
        can be used to set the minimum eigenvalue of the Z matrix products. This
        can be useful when the data is very noisy and leads to non-positive-definite
        matrices. The `clip_z` parameter represents the minimum allowed eigenvalue
        of the Z matrix when normalized so that the maximum eigenvalue is 1.0. 
        In this way, you cap the maximum condition number for all Z products 
        to 1/clip_z. This generally should be kept near machine precision. If it 
        is needed, try setting this to a small value i.e. 1e-16 first before 
        increasing it. Setting this value to None will disable the clipping, and 
        should always be the default option unless you are experiencing issues 
        like NaNPairwiseError.

        Args:
            psrs (list): A list of `enterprise.pulsar.BasePulsar` objects.
            pta (enterprise.signals.signal_base.PTA): A PTA object.
            gwb_name (str): The name of the GWB in the PTA object. Defaults to 'gw'.
            corepath (str, optional): The location of a pre-saved Core object.
            core (la_forge.core.Core, optional): A `la_forge.core.Core` object.
            chain_path (str, optional): The location of an PTMCMC chain.
            chain (np.ndarray, optional): The sampled chain.
            param_names (str or list, optional): The names of the chain parameters.
            orfs (list, optional): An orf name or function or list of orf names
                or functions. See `set_orf()`. Defaults to ['hd'].
            orf_names (str or list, optional): The names of the corresponding orfs.
                See `set_orf()`. Set to None for default names.
            pcmc_orf (str or function, optional): The assumed ORF for the pair
                covariance matrix when using the MCOS. See `set_orf()`.
                Defaults to None.
            clip_z (float, optional): (Experimental) The minimum eigenvalue of the 
                Z matrix products. Can be useful with very noisy data. Set to None 
                for no clipping. See doc comments for details. Defaults to None.
            sub_tqdm (bool): Whether to use progress bars on frequency and
                pair covariance calculations. Defaults to False.
            pre_cache (bool): Whether to pre-cache the many matrix products.
                Note that this only moves the waiting time to the initialization
                instead of the first calls. Defaults to True.

        Raises:
            TypeError: If the PTA object is not of type `enterprise.signals.signal_base.PTA`.
            TypeError: If the pulsars in the psrs list are not a list or of type 
                `enterprise.pulsar.BasePulsar`.
        """
        self.sub_tqdm = sub_tqdm # Additional progress bars if needed

        if isinstance(pta, ent_sig.signal_base.PTA): # Check that pta is a PTA object
            self.pta = pta
        else:
            raise TypeError("pta supplied is not of type 'enterprise.signals.signal_base.PTA'!")
        self.psr_signals = self.pta._signalcollections # I don't like typing this a million times

        # Set the GWB name
        self.gwb_name = gwb_name # Assume that the GWB name is right
        # Lets also get the GW signal to calculate the GWB phi. Assumes that the spectrum
        # is identical for all pulsars.
        gw_sig = [s for s in self.psr_signals[0].signals if s.signal_id==self.gwb_name][0]
        self._get_phi = gw_sig.get_phi

        self.psrs = psrs # Duck typing! Assume that the psrs behave like Enterprise pulsars
        try: 
            _ = psrs[0] 
        except TypeError: 
            raise TypeError("psrs list supplied is not able to be indexed")
        
        # Check for marginalizing timing model (only checks the first pulsar)
        self.marginalizing_timing_model = False
        for s in self.psr_signals[0].signals:
            if 'marginalizing linear timing model' in s.signal_name:
                self.marginalizing_timing_model = True

        # Set up method caching!
        self._cache_params = None
        self.npsr = len(psrs)
        set_cache_params(self, self.pta) # Sets self._cache_params
        
        # Set the core and chain parameters (if provided)
        self.lfcore, self.max_like_params, self.max_like_idx = None, None, None
        self.set_chain_params(core, core_path, chain_path, chain, param_names)

        # Get the frequencies of the PTA
        self.freqs = utils.get_pta_frequencies(pta, gwb_name)
        self.nfreq = len(self.freqs) 
        
        # Set the pulsar pair indices and names
        self.pair_idx = np.array(np.triu_indices(self.npsr,1)).T # [n_pairs, 2]
        self.pair_names = [(self.psrs[a].name,self.psrs[b].name) for a,b in self.pair_idx]
        self.npairs = len(self.pair_names)

        # Set up ORF attributes
        self.norfs, self.orf_functions = 0, []
        self.orf_design_matrix, self.orf_matrix, self.mcos_orf = None, None, None
        self.orf_names = None
        self.set_orf(orfs, orf_names, pcmc_orf)
        self.nside, self.lmax = None, None 

        # Some extra attributes used by various things
        self.nmos_iterations = {} # Used to store the NMOS iterations mid-way through
        self.max_matrix_chunk = 300 # Users can set after creation if they want to change it

        if pre_cache:
            iterable = range(self.npsr)
            if self.sub_tqdm:
                iterable = tqdm(iterable, desc='Pre-caching matrix products')
            # Pre-cache matrix products which do not vary
            for i in iterable:
                # Check if this pulsar has model dependent parameters
                has_wn = len(self._cache_params['white_noise'][i]) > 0
                has_basis = len(self._cache_params['basis'][i]) > 0
                has_delay = len(self._cache_params['delay'][i]) > 0

                # Must not have white noise or basis parameters
                if (not has_wn) and (not has_basis): 
                    # Can cache things!
                    _ = self._get_FNF(i,params={})
                    _ = self._get_TNT(i,params={})
                    _ = self._get_FNT(i,params={})

                    if not has_delay:
                        # Can cache dt products as well
                        _ = self._get_FNdt(i,params={})
                        _ = self._get_TNdt(i,params={})

        # Experimental stuff ---------------------------------------------------
        self.clip_z = clip_z
        if clip_z is not None:
            warn("Clipping Z matrix products is an experimental feature. Use with caution.")
        

    def set_chain_params(self, core=None, core_path=None, chain_path=None, 
                         chain=None, param_names=None):
        """A method to add MCMC chains to an OptimalStatistic object. 

        This method takes a number of different forms of MCMC chains and creates
        a `la_forge.core.Core` object for use with noise marginalization or maximum
        likelihood optimal statistic. To use this method, you must include one
        or more of the following options from most prefered to least:
            1. `core`
            2. `corepath`
            3. `chain_path`
            4. `chain` & `param_names`
            5. `chain`

        The core object can then be accessed through `self.lfcore` with the maximum
        likelihood parameters stored in `self.max_like_params` or through its chain
        index in `self.max_like_idx`.

        Args:
            core (la_forge.core.Core, optional): A `la_forge.core.Core` object. 
            corepath (str, optional): The path to a saved `la_forge.core.Core` object. 
            chain_path (str, optional): The path to a saved chain from PTMCMC. 
            chain (numpy.ndarray, optional): The raw numpy.ndarray of the MCMC chain. 
            param_names (list, optional): The order of the parameter names of chain. 
        """
        # Prefered order for loading chains: 
        # core > corepath > chain_path > chain + param_names > chain
        if core is not None and isinstance(core, Core):
            self.lfcore = core # Easy peasy! Duck typing!
        elif core_path is not None:
            self.lfcore = Core(corepath=core_path) # Load core from path
        elif chain_path is not None:
            self.lfcore = Core(chaindir=chain_path) # Load chain from path
        elif chain is not None and param_names is not None:
            # Assume that the chain ordering is the same as the parameter names
            # since we have no way to check!
            self.lfcore = Core(chain=chain,params=param_names) 
        elif chain is not None:
            # Assume that the chain ordering is the same as the PTA parameter names
            # again, we can't check!
            self.lfcore = Core(chain=chain,params=self.pta.param_names)
        else:
            return # Nothing supplied!

        # Set the maximum likelihood parameters
        params, idx = utils.get_max_like_params(self.lfcore)
        self.max_like_params = params
        self.max_like_idx = idx


    def get_chain_params(self, N=1, idx=None, format='dict', freespec_fix=False):
        """A method to get samples from the `self.lfcore` object.

        This method is a helper method to either get random samples from the chain
        or to get specific samples if `idx` is supplied. This method can be useful
        when you want to know which parameters were used in a particular iteration
        of the noise marginalizing process. This can especially be helpful with 
        varied gamma CURN models. 

        If `idx` is None then this method will return N random samples from the chain.
        If `idx` is either an int or an array of ints, then this method will return
        the samples at those specific indexes. 
        NOTE: `idx` refer to the unburned indexes of the chain. (i.e. idx=0 will
        be the first sample in the chain, which will be in the burn-in.)
        NOTE: N is ignored if `idx` is supplied.

        The `format` parameter can be set to either 'dict' or 'array'. If 'dict' is
        chosen, then the samples will be returned as a list of dictionaries where the
        keys are the parameter names. If 'array' is chosen, then the samples will be
        returned as a numpy.ndarray.
        NOTE: If using 'array' to get the parameters associated with each
        index i in array[:,i] you can use `self.lfcore.params`.

        The `freespec_fix` flag is to fix a minor bug in enterprise which
        expects the freespec GWB parameters to be of the form 'gw_log10_rho' instead
        of the sampled 'gw_log10_rho_0', 'gw_log10_rho_1', .... Enabling this
        flag will add an additional key value pair to the dictionary return. 
        NOTE: This will only be applied to the 'dict' format.

        Args:
            N (int, optional): The number of random samples to generate. Defaults to 1.
            idx (int or np.ndarray, optional): The index or indexes of the samples to return.
                Set to None for random samples. Defaults to None.
            format (str): The format of the return. Can be 'dict' or 'array'.
                Defaults to 'dict'.
            freespec_fix (bool): Whether to fix the freespec GWB parameter names 
                to make Enterprise happy. Defaults to False.
            
        Raises:
            ValueError: If the chain parameters are not set. Set with `set_chain_params()`.
            ValueError: If the format is not either 'dict' or 'array'.

        Returns:
            dict: A dictionary with keys:
                - 'samples': A list of samples from the chain. [N x n_params]
                (May be a list of dictionaries if `format` is 'dict')
                - 'idx': A numpy array of indexes of the samples. [N] 
        """

        if self.lfcore is None:
            msg = 'No chain parameters set! Set these before calling this method.'
            raise ValueError(msg)
        
        if idx is None:
            # If indexes are not supplied, then choose them randomly without burn-in
            idx = np.random.randint(self.lfcore.burn, self.lfcore.chain.shape[0], N)
        else:
            if isinstance(idx, (int, np.integer)): # Sometimes idx is a single int!
                idx = [idx] # put it in a list so np.array() is happy
            idx = np.array(idx) # into an array of indices

        if format == 'array':
            # Return samples as a numpy array
            samples = self.lfcore.chain[idx]

        elif format == 'dict':
            # Return a list of dictionaries
            samples = []
            for i in idx: # Not sure if there is a better way than just a for loop
                d = {p:v for p,v in zip(self.lfcore.params, self.lfcore.chain[i])}

                if freespec_fix:
                    # Adds the 'gw_log10_rho' parameter to the dictionary (from individual)
                    d = utils.freespec_param_fix(d, self.gwb_name)

                samples.append(d)

        else:
            msg = f"Format {format} not recognized. Use 'dict' or 'array'."
            raise ValueError(msg)
        
        return {'samples': samples, 'idx': idx}
    

    def _parse_params(self, params, N, gamma):
        """A helper method to parse the parameters for the OS and PFOS.

        This method is hidden from the user and is not meant to be called directly.

        This method is used to parse the parameters for the OS and PFOS. This method
        is used to normalize the parameters used in these methods while still allowing
        the user to use a variety of different usage formats.

        'params' can be either a dictionary, a list of dictionaries, a list of indexes,
        or a single index. 

        'N' is the number of random parameter vectors to generate. This is only used
        if 'params' is None. 

        'gamma' is the spectral index to use for the analysis. If 'gamma' is a float,
        then the method will use that value for all iterations. If 'gamma' is an iterable,
        then the method will use each value for each iteration. If 'gamma' is None, 
        then the method will first check if gamma is in the parameters, otherwise it will
        assume that the PTA model is a fixed gamma and take it from there.

        If no indexes are found (i.e. specific parameter vectors are used), then
        this method with return -1 as the index for those iterations.

        Args:
            params (dict or list, optional): The parameters or indexes to use. Check 
                the documentation for usage info. Defaults to None.
            N (int, optional): The number of random parameter vectors to generate. Only used
                if params is None. Defaults to 1.
            gamma (float or list, optional): The spectral index to use for the analysis. Check
                documentation for usage info. Defaults to None.

        Raises:
            ValueError: If params is not a dictionary, list of dictionaries, or list of indexes.
            ValueError: If gamma is not a float or a list of floats.
            ValueError: If gamma is not the same length as the number of iterations.
            ValueError: If the parameters are not able to be determined.

        Returns:
            tuple: A tuple of the parameters, indexes, and gamma values for each iteration.
        """
        
        # Get the parameters in the format of a list of dictionaries------------
        if params is not None:
            # Reformat so that we always have a list of dictionaries
            if isinstance(params, dict):
                # A single dictionary of parameters
                idx = [-1]
                pars = [utils.freespec_param_fix(params, self.gwb_name)]
            elif hasattr(params, '__iter__') and isinstance(params[0], dict):
                # An iterable of dictionaries
                idx = [-1]*len(params)
                pars = [utils.freespec_param_fix(p, self.gwb_name) for p in params]
            elif hasattr(params, '__iter__') and isinstance(params[0], (int, np.integer)):
                # An iterable of indexes
                idx = params
                out = self.get_chain_params(idx=params, format='dict', freespec_fix=True)
                pars = out['samples']
            elif isinstance(params, (int, np.integer)):
                # A single index
                idx = [params]
                out = self.get_chain_params(idx=params, format='dict', freespec_fix=True)
                pars = out['samples']
            else:
                msg = "params must be a dictionary, list of dictionaries, or list of indexes."
                raise ValueError(msg)
        elif N==1: # params is None and N==1
            # Default to maximum likelihood
            idx = [self.max_like_idx]
            pars = [utils.freespec_param_fix(self.max_like_params, self.gwb_name)]
        elif N>1: # params is None and N>1
            # Default to noise marginalization
            out = self.get_chain_params(N, format='dict', freespec_fix=True)
            pars, idx = out['samples'], out['idx']
        else:
            msg = "Unable to determine parameters. Set either params or N>=1."
            raise ValueError(msg)
        
        # Deal with gamma values -----------------------------------------------
        if gamma is not None:
            # Set gamma, check if list or single value
            if hasattr(gamma, '__iter__'):
                gam = gamma
            elif isinstance(gamma, (float, np.floating)):
                gam = [gamma]*len(pars)
            else:
                msg = "gamma must be a float or a list of floats."
                raise ValueError(msg)
            # Check if gamma is the right length
            if len(gam) != len(pars):
                msg = "gamma must be a single float or the same length as the number of iterations."
                raise ValueError(msg)
        elif self.gwb_name+'_gamma' in pars[0]:
            # Gamma is none, check if it is in the parameters
            gam = [p[self.gwb_name+'_gamma'] for p in pars]
        else:
            # Gamma is none and not in the parameters, get fixed gamma if available
            gam = utils.get_fixed_gwb_gamma(self.pta, self.gwb_name)
            gam = [gam]*len(pars)

        return pars, idx, gam


    def set_orf(self, orfs=['hd'], orf_names=None, pcmc_orf=None):
        """Sets the overlap reduction function[s] (ORF) for the cross correlations.

        Sets the overlap reduction function[s] of the cross correlation and the
        corresponding ORF design matrix. This method supports multiple ORFs by
        simply supplying a list of ORFs. `orf_names` can be left as None to use default
        names. `orfs` can also be a user-defined function which accepts 2
        `enterprise.pulsar.BasePulsar` objects.

        Otherwise, use one of the following pre-defined ORF within 
        `defiant.orf_functions.defined_orfs`:
            - 'hd' or 'hellingsdowns': Hellings and Downs
            - 'dp' or 'dipole': Dipole
            - 'mp' or 'monopole': Monopole
            - 'gwdp' or 'gw_dipole': Gravitational wave dipole
            - 'gwmp' or 'gw_monopole': Gravitational wave monopole
            - 'st' or 'scalar_tensor': Scalar tensor
            - 'l_' or 'legendre_': Legendre polynomials where the number after 
                    the _ is the degree
            
        **Experimental**: `pcmc_orf` is an experimental feature that aims to curb 
        the problematic nature of pair covariance matrix with the MCOS. In order 
        to construct the pair covariance matrix, an assumed ORF must be used. 
        With a single ORF, the answer is trivial, but with multiple ORFs, the ratio 
        of power for each component is not known a priori. This can lead to large 
        changes in the covariance matrix for minor changes in the assumed power in 
        each component. The `pcmc_orf` parameter allows the user to set an assumed ORF
        which will override the default behavior of the MCOS. This can be especially
        useful when using many ORFs or components of the ORF are expected to be
        near zero.

        If `pcmc_orf` argument is set to None (default): 
            - The typical behavior will be used, this being that the power per 
            process will be the normalized non-pair covariant MCOS multiplied 
            by the CURN amplitude. 
        If `pcmc_orf` argument is set to a string: 
            - The pair covariance matrix will be computed using a singular ORF 
            function specified (i.e. 'hd').
        If `pcmc_orf` argument is set to a function:
            - The pair covariance matrix will be computed using the user supplied 
            function. This function must accept two `enterprise.pulsar.BasePulsar`
            objects as inputs and outputs a float for their ORF.

        
        Args:
            orfs (str or function or list): An ORF string, function or list of 
                strings and/or functions. Note that a custom function must 
                accept two `enterprise.pulsar.BasePulsar` objects as inputs 
                and outputs a float for their ORF.
            orf_names (list, optional): The names of the corresponding orfs. Set to None
                for default names.
            pcmc_orf (str or function, optional): The assumed ORF for the pair covariance 
                    matrix. when using the MCOS. Defaults to None.

        Raises:
            ValueError: If the length of `orfs` and `orf_names` does not match.
            ORFNotFoundError: If a pre-defined ORF is not found.
            TypeError: If the user-supplied ORF does not have correct format. 
        """
        # Need to check if orfs is a string, list, or function
        if not hasattr(orfs, '__iter__') or isinstance(orfs, str):
            orfs = [orfs] # Make it a list if it is not

        # Check if orf_names is None or a string
        if orf_names is None:
            orf_names = [None for a in orfs]
        elif isinstance(orf_names, str):
            orf_names = [orf_names]

        # Check for same length!
        if len(orfs) != len(orf_names):
            msg = 'length of orfs and length of orf_names is not equal!'
            raise ValueError(msg)
        
        self.norfs = len(orfs)
        self.orf_names = [] # Will be filled with names later

        # Used internal and external
        self.orf_design_matrix = np.zeros( (self.norfs, self.npairs) ) 
        # Used internal for pair covariance
        self.orf_matrix = np.ones( (self.norfs, self.npsr, self.npsr) ) 
        self.orf_functions = []

        for i in range( len(orfs) ): # Hence why we listified orfs
            orf = orfs[i]
            name = orf_names[i]

            if isinstance(orf, str):
                # ORF must be pre-defined function
                # Get the ORF function from orf_functions.py
                cur_orf = orf_funcs.get_orf_function(orf) 
                # Set the name of the ORF if user did not provide one
                name = orf if name is None else name
                
                self.orf_names.append(name)
                self.orf_functions.append(cur_orf)

                # Compute the ORF design matrix and ORF matrix
                for j,(a,b) in enumerate(self.pair_idx): 
                    # For each pair of pulsars
                    v = cur_orf(self.psrs[a] , self.psrs[b]) 
                    self.orf_design_matrix[i,j] = v
                    self.orf_matrix[i,a,b] = v
                    self.orf_matrix[i,b,a] = v

            else:
                # ORF is user supplied function
                # If the user did not provide a name, use the function name
                name = orf.__name__ if name is None else name 

                self.orf_names.append(name)
                self.orf_functions.append(orf)
                try:
                    for j,(a,b) in enumerate(self.pair_idx):
                        # For each pair of pulsars
                        v = orf(self.psrs[a], self.psrs[b])
                        self.orf_design_matrix[i,j] = v
                        self.orf_matrix[i,a,b] = v
                        self.orf_matrix[i,b,a] = v
                except Exception as e:
                    msg = f"Cannot use custom ORF function {orf}. Ensure that " +\
                           "the function has two parameters, both of which accept "+\
                           "the 'enterprise.pulsar.BasePulsar' as types."
                    raise TypeError(msg) from e
            
        # We need to make sure that it has (n_pairs x n_orfs), 
        # but we made it as (n_orf x n_pairs).
        self.orf_design_matrix = self.orf_design_matrix.T

        # Additional bits for PC+MC
        if pcmc_orf is not None:
            if isinstance(pcmc_orf, str):
                # Pre-defined ORF
                orf = orf_funcs.get_orf_function(pcmc_orf)
            else:
                # User designed ORF
                orf = pcmc_orf

            temp = np.zeros( (self.npsr,self.npsr) )
            for a in range(self.npsr):
                for b in range(a+1,self.npsr):
                    v = orf(self.psrs[a],self.psrs[b])
                    temp[a,b] = v
                    temp[b,a] = v
            self.mcos_orf = temp
        else:
            # Not using pcmc ORF, so set to None
            self.mcos_orf = None


    def set_anisotropy_basis(self, basis='pixel', nside=2, lmax=6, pc_orf='hd'):
        """A method to set the anisotropy basis for the OS.

        This method sets the anisotropy basis for the OS. `basis` can be either
        a 'pixel' basis or a 'spherical' harmonic basis. The pixel basis is a simple
        pixelization of the sky, while the spherical harmonic uses a parameterized
        version of the pixel basis. If the basis is set to 'pixel', lmax is
        ignored. For pair covariance to work correctly, an assumed ORF should be set,
        which is done with the `pc_orf` argument. This must be a string or custom
        ORF function. The predefined ORFs can be found in `orf_functions.defined_orfs`.
        `pc_orf` works identically to `pcmc_orf` in `set_orf()`.

        Args:
            basis (str): The basis for the anisotropy. Must be 'pixel' or 'spherical'.
            nside (int): The nside of the pixelization. Defaults to 2.
            lmax (int, optional): The maximum l value of the spherical harmonics. Defaults to 6.
            pc_orf (str): The ORF to use for the pair covariance matrix. Defaults to 'hd'.
        """
        # Handle the ORF for pair covariance
        if isinstance(pc_orf, str):
            # Pre-defined ORF
            orf = orf_funcs.get_orf_function(pc_orf)
        else:
            # User designed ORF
            orf = pc_orf

        # Set the ORF matrix for pair covariance
        orf = orf_funcs.get_orf_function(pc_orf)
        temp = np.zeros( (self.npsr,self.npsr) )
        for a in range(self.npsr):
            for b in range(a+1,self.npsr):
                v = orf(self.psrs[a],self.psrs[b])
                temp[a,b] = v
                temp[b,a] = v
        self.mcos_orf = temp

        # Set the anisotropy basis
        if basis.lower() == 'pixel':
            basis = orf_funcs.anisotropic_pixel_basis(self.psrs, nside, self.pair_idx)

            self.orf_names = [f'pixel_{i}' for i in range(basis.shape[1])]
            self.orf_design_matrix = basis
            self.orf_matrix = None
            self.norfs = basis.shape[1]

        elif basis.lower() == 'spherical':
            basis = orf_funcs.anisotropic_spherical_harmonic_basis(self.psrs, lmax, 
                                                                nside, self.pair_idx)

            self.orf_names = [f'c_{l},{m}' for l in range(lmax+1) for m in range(-l,l+1)]
            self.orf_design_matrix = basis
            self.orf_matrix = None
            self.norfs = basis.shape[1]
    
        else:
            raise os_ex.ORFNotFoundError(f"Anisotropy basis {basis} not found!")
        

    def compute_OS(self, params=None, N=1, gamma=None, pair_covariance=False, 
                   return_pair_vals=True, fisher_diag=False, use_tqdm=True):
        """Compute the OS and its various modifications.

        This is one of 2 main methods of the `OptimalStatistic` class. This method
        can compute any flavor of the OS which uses broadband estimation (i.e. constructs
        a single estimator for the whole spectrum). There are many forms in which you can
        use this method, and checking the decision tree is best for determining exactly
        what you might want and what parameters to set to accomplish that. 
        NOTE: Since this method's outputs can vary widely, it will return in a dictionary
        with the keys being the output names.

        The basic usage of this method can be boiled down to the following:
        If you want to compute a single iteration of the OS:
            - supply a dictionary/array of `params`. By default, if `params=None`
              and `N=1`, this will compute the maximum likelihood OS.
        If you want to compute the noise marginalized OS:
            - supply a list of dictionaries for `params`
            - ensure that a La forge core set (see `set_chain_params()`) 
              and set `params=None` and `N>1`.
        If you want to compute the OS with pair covariance:
            - simply set pair_covariance=True. This will also replace the pulsar
              pair covariance matrix, C, that gets returned.
        If you want to change the spectral index of the GWB model, you can either:
            - set a particular gamma value for all NMOS iterations by setting `gamma` to a float
            - set `gamma=None` and the method will default to each iterations' gamma value
              or the fixed gamma value if using a fixed gamma model.

        Users can choose to forgo returning the pairwise values (i.e. xi, rho, sig, C)
        when executing this method by setting `return_pair_vals=False`. This can be useful
        when you are only interested in the OS estimators and wish to save memory as
        the pairwise covariance matrices can be very large. 

        The `params` argument also has the additional functionality to allow users
        to supply specific parameter index or indexes by setting `params` to an int or
        a list of ints.

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component OS
        processes simultaneously. Keep this `True` unless you know you want this.

        Args:
            params (dict, list, int, optional): The parameters to use in the OS. 
                Users can supply any of the following for this argument:
                    - dict: A dictionary of parameter values
                    - list: A list of dictionaries of parameter values
                    - int: An index of a parameter vector in the chain
                    - list: A list of indexes of parameter vectors in the chain
                    - None: If `N=1`, defaults to maximum likelihood.
                    - None: If `N>1`, use random samples from the chain. (NMOS)
            N (int): The number of NMOS iterations to run. If `params` is not None,
                this is argument ignored.
            gamma (float, optional): The spectral index to use for analysis. If set to None,
                this method first checks if gamma is in `params`, otherwise it will
                assume the PTA model is a fixed gamma and take it from there. Defaults to None.
            pair_covariance (bool): Whether to use pulsar pair covariance. Defaults to False.
            return_pair_vals (bool): Whether to return the `xi`, `rho`, `sig`, `C` values. 
                Defaults to True.
            fisher_diag (bool): Whether to zero the off-diagonal elements of the
                fisher matrix. Defaults to False.
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.

        Returns:
            dict: A dictionary containing several keys:
                - 'A2': The OS amplitude estimators at 1/yr [N x Norfs]
                - 'A2s': The 1-sigma uncertainties of A2 if norfs==1 [N]
                  or a covariance matrix on the fits if norfs>1 [N x Norfs x Norfs]
                - 'idx': The index(es) of the parameter vectors used in NM 
                  each iteration. If no indexes are found indexes will be -1 [N]

            If `return_pair_vals` is True:
                - 'xi': The pair separations of the pulsars [npairs]
                - 'rho': The pair correlated powers [N x npairs]
                - 'sig': The pair uncertainties in rho [N x npairs]
                - 'C': The pair covariance matrix [N x npairs x npairs]
        """
        # First, get the parameter dictionaries
        all_pars, all_idx, all_gamma = self._parse_params(params, N, gamma)

        # Now setup the return values inside of the self.nmos_iterations
        self.nmos_iterations = {'A2':[],'A2s':[],'idx':[]}
        if return_pair_vals:
            # Compute the pulsar pair separations and initialize other values
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self.pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rho'] = []
            self.nmos_iterations['sig'] = []
            self.nmos_iterations['C'] = []   

        iterable = range(len(all_pars))
        if use_tqdm and len(all_pars)>1: # Add tqdm if more than 1 iteration
            iterable = tqdm(iterable, desc='NMOS Iters')

        # Noise marginalized loop
        for i in iterable:
            par, idx, gam = all_pars[i], all_idx[i], all_gamma[i]
            phihat = utils.powerlaw(self.freqs, 0, gam, modes=2) # 2 modes: Sine and Cosine

            X, Z = self.compute_XZ(par)

            # Now compute the pairwise correlations and uncertainties
            rho_ab, sig_ab = self.compute_rho_sig(X, Z, phihat)

            # Compute the covariance matrix
            if pair_covariance:
                method = 'woodbury' # Pair covariance can utilize woodbury identity!

                # Need some estimate of A2 to compute the covariance matrix
                a2_est = utils.get_a2_estimate(par, self.freqs, gam, self.gwb_name, self.lfcore)

                # Compute the pair covariance matrix: 3 behaviors
                if self.norfs==1:
                    # Single ORF (phi = A2*phihat)
                    C = pc.create_OS_pair_covariance(Z, phihat, a2_est*phihat, self.orf_matrix[0], 
                            sig_ab**2, use_tqdm and self.sub_tqdm, self.max_matrix_chunk)
                        
                elif self.norfs>1 and self.mcos_orf is not None:
                    # Assumed ORF with MCOS
                    C = pc.create_OS_pair_covariance(Z, phihat, a2_est*phihat, self.mcos_orf,
                            sig_ab**2, use_tqdm and self.sub_tqdm, self.max_matrix_chunk)
                        
                else:
                    # Default behavior
                    C = pc.create_MCOS_pair_covariance(Z, phihat, self.orf_matrix,
                            self.orf_design_matrix, rho_ab, sig_ab, a2_est, 
                            use_tqdm and self.sub_tqdm, self.max_matrix_chunk)

            else:
                method = 'diagonal'
                C = np.diag(sig_ab**2)

            # Now that we have the data, model, and covariance, we can compute!
            s = np.diag(sig_ab**2)

            A2, A2s = utils.linear_solve(self.orf_design_matrix, C, rho_ab, s, 
                                         method, fisher_diag)

            A2 = np.squeeze(A2) if self.norfs>1 else A2.item()
            A2s = np.squeeze(A2s) if self.norfs>1 else np.sqrt(A2s.item())

            self.nmos_iterations['A2'].append(A2)
            self.nmos_iterations['A2s'].append(A2s)
            self.nmos_iterations['idx'].append(idx)
            if return_pair_vals:
                self.nmos_iterations['rho'].append(rho_ab)
                self.nmos_iterations['sig'].append(sig_ab)
                self.nmos_iterations['C'].append(C)
        # End of the NMOS loop

        # Setup our return values
        output = {}
        output['A2'] = np.squeeze( self.nmos_iterations['A2'] ) # [N x Norfs]
        output['A2s'] = np.squeeze( self.nmos_iterations['A2s'] ) # [N x Norfs x Norfs]
        output['idx'] = np.squeeze( self.nmos_iterations['idx'] ) # [N]

        if return_pair_vals:
            output['xi'] = np.squeeze( self.nmos_iterations['xi'] ) # [npairs]
            output['rho'] = np.squeeze( self.nmos_iterations['rho'] ) # [N x npairs]
            output['sig'] = np.squeeze( self.nmos_iterations['sig'] ) # [N x npairs]
            output['C'] = np.squeeze( self.nmos_iterations['C'] ) # [N x npairs x npairs]
        
        self.nmos_iterations = {} # Empty the dictionary to save memory
        return output


    def compute_PFOS(self, params=None, N=1, pair_covariance=False, narrowband=False, 
                     return_pair_vals=True, fisher_diag=False, select_freq=None, use_tqdm=True):
        """Compute the PFOS and its various modifications.

        This is one of 2 main methods of the `OptimalStatistic` class. This method
        can computes the different flavors of the PFOS (i.e. a free-spectrum search). 
        There are many forms in which you can use this method, and checking the 
        decision tree is best for determining exactly what you might want and what 
        parameters to set to accomplish that. 
        NOTE: Since this method's outputs can vary widely, it will return in a dictionary
        with the keys being the output names.
        
        The basic usage of this method can be boiled down to the following:
        If you want to compute a single iteration of the PFOS:
            - supply a dictionary of `params`. By default, if `params=None` and `N=1`, 
              this will compute the maximum likelihood PFOS.
        If you want to compute the noise marginalized PFOS:
            - supply a list of dictionaries for `params`
            - ensure that a La forge core set (see `set_chain_params()`) 
              and set `params=None` and `N>1`. 
        If you want to compute the PFOS with pair covariance:
            - simply set `pair_covariance=True`. This will also replace the covariance 
              matrix, `C`, that gets returned.

        The `params` argument also has the additional functionality to allow users
        to supply specific parameter index or indexes by setting `params` to an int or
        a list of ints.

        Users can choose to forgo returning the pairwise values (i.e. `xi`, `rhok`, 
        `sigk`, `Ck`) when executing this method by setting `return_pair_vals=False`. 
        This can be useful when you are only interested in the PFOS estimators and 
        wish to save memory as the pairwise covariance matrices can be very large. 

        The `narrowband` flag is used to compute the narrowband-normalized PFOS
        detailed in Gersbach et al. 2024. This makes an additional simplification
        to the PFOS at the cost of accuracy in detectable GWB signals. This is mostly
        included for legacy and bugfixing purposes.

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component PFOS
        processes simultaneously. Keep this on True unless you know what you are doing.

        If you want to select a specific frequency to compute the PFOS at, you can set
        the `select_freq` parameter to the index of the frequency you wish to compute the
        PFOS at. This can be useful if you only care about a specific frequency. Setting
        `select_freq=None` will compute the PFOS at all frequencies.

        Args:
            params (dict, list, int, optional): The parameters to use in the PFOS. 
                Users can supply any of the following for this argument:
                    - dict: A dictionary of parameter values
                    - list: A list of dictionaries of parameter values
                    - int: An index of a parameter vector in the chain
                    - list: A list of indexes of parameter vectors in the chain
                    - None: If `N==1`, defaults to maximum likelihood.
                    - None: If `N>1`, use random samples from the chain. (NM PFOS)
            N (int): The number of NM PFOS iterations to run. If `params` is not None,
                this is argument ignored.
            narrowband (bool): Whether to use the narrowband-normalized PFOS instead of
                the default broadband-normalized PFOS. Defaults to False.
            pair_covariance (bool): Whether to use pair covariance. Defaults to False.
            return_pair_vals (bool): Whether to return the `xi`, `rhok`, `sigk`, `Ck` 
                values. Defaults to True.
            fisher_diag (bool): Whether to zero the off-diagonal elements of the
                fisher matrix. Defaults to False.
            select_freq (int, optional): The index of the frequency to compute the PFOS at.
                Defaults to None (all frequencies).
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.

        Returns:
            dict: A dictionary containing several keys:
                - 'Sk': The PFOS PSD/Tspan estimators at each frequency [N x nfreq x Norfs]
                - 'Sks': The 1-sigma uncertainties of Sk if norfs==1 [N x nfreq]
                  or a covariance matrix on the fits if norfs>1 [N x nfreq x Norfs x Norfs]
                - 'idx': The index(es) of the parameter vectors used in NM 
                  each iteration. If no indexes are found indexes will be -1 [N]

            If `return_pair_vals` is True:
                - 'xi': The pair separations of the pulsars [npairs]
                - 'rhok': The pair correlated powers per frequency [N x nfreq x npairs]
                - 'sigk': The pair uncertainties in rhok [N x nfreq x npairs]
                - 'Ck': The pair covariance matrix per frequency [N x nfreq x npairs x npairs]
        """
        # First, get the parameter dictionaries. Set gamma, but ignore it anyway
        all_pars, all_idx, _ = self._parse_params(params, N, 1.0)

        # Now setup the return values inside of the self.nmos_iterations
        self.nmos_iterations = {'Sk':[],'Sks':[],'idx':[]}
        if return_pair_vals:
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self.pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rhok'] = []
            self.nmos_iterations['sigk'] = []
            self.nmos_iterations['Ck'] = []  

        # Create iterable for the NMOS iterations
        iterable = range(len(all_pars))
        if use_tqdm and len(all_pars)>1:
            iterable = tqdm(iterable, desc='NMOS Iters')


        for i in iterable: # NMOS loop
            par, idx = all_pars[i], all_idx[i]
            # Get the handy OS matrix products
            X,Z = self.compute_XZ(par)

            # Need the approximate shape of the GWB spectrum, use CURN GWB
            phi = self._get_phi(par)

            # All frequencies if not using a selected frequency
            freq_iterable = range(self.nfreq)
            fk = range(self.nfreq)
            if select_freq is not None: # Selected a frequency
                freq_iterable = [0] # Only one frequency
                fk = [select_freq] # But not at k=0
            
            # Add tqdm to the frequency iterable if sub_tqdm is set or only 1 iteration
            if (self.sub_tqdm or len(iterable)==1) and len(freq_iterable)>1 and use_tqdm:
                freq_iterable = tqdm(freq_iterable, desc='PFOS Freqs')

            Sk, Sks = [],[]
            if return_pair_vals:
                rhok, sigk, Ck = [], [], []

            for j,k in zip(freq_iterable,fk):
                # Get correlated psd, uncertainty, and normalization
                rho, sig, norm = self.compute_rhok_sigk(X, Z, phi, k, narrowband)

                if pair_covariance:
                    method = 'woodbury' # Pair covariance can utilize woodbury identity!
                    if self.norfs==1:
                        # Single ORF
                        C = pc.create_PFOS_pair_covariance(Z, phi, self.orf_matrix[0], 
                                norm, narrowband, k,
                                use_tqdm and self.sub_tqdm, self.max_matrix_chunk)
                    
                    elif self.norfs>1 and self.mcos_orf is not None:
                        # Assumed ORF with MCOS
                        C = pc.create_PFOS_pair_covariance(Z, phi, self.mcos_orf, 
                                norm, narrowband, k, 
                                use_tqdm and self.sub_tqdm, self.max_matrix_chunk)
                    
                    else:
                        # Default behavior
                        C = pc.create_MCPFOS_pair_covariance(Z, phi, self.orf_matrix, 
                                self.orf_design_matrix, rho, sig, 
                                norm, narrowband, k,
                                use_tqdm and self.sub_tqdm, self.max_matrix_chunk)
                    
                else:
                    method = 'diagonal'
                    C = np.diag(sig**2)

                # Now that we have the data, model, and covariance, we can compute!
                s_diag = np.diag(sig**2)
                A, S = utils.linear_solve(self.orf_design_matrix, C, rho, 
                                          s_diag, method, fisher_diag)
                                
                Sk.append( np.squeeze(A) )
                Sks.append( np.squeeze(S) if self.norfs>1 else np.sqrt(S.item()) )
                if return_pair_vals:
                    rhok.append( np.squeeze(rho) )
                    sigk.append( np.squeeze(sig) )
                    Ck.append( np.squeeze(C) )

            # End of frequency loop
            self.nmos_iterations['Sk'].append( np.squeeze(Sk) )
            self.nmos_iterations['Sks'].append( np.squeeze(Sks) )
            self.nmos_iterations['idx'].append(idx)
            if return_pair_vals:
                self.nmos_iterations['rhok'].append( np.squeeze(rhok) )
                self.nmos_iterations['sigk'].append( np.squeeze(sigk) )
                self.nmos_iterations['Ck'].append( np.squeeze(Ck) )

        # End of NMOS loop
        # Setup our return values
        output = {}
        output['Sk'] = np.squeeze( self.nmos_iterations['Sk'] ) # [N x nfreq x Norfs]
        output['Sks'] = np.squeeze( self.nmos_iterations['Sks'] ) # [N x nfreq x Norfs x Norfs]
        output['idx'] = np.squeeze( self.nmos_iterations['idx'] ) # [N]
        if return_pair_vals:
            output['xi'] = np.squeeze( self.nmos_iterations['xi'] ) # [npairs]
            output['rhok'] = np.squeeze( self.nmos_iterations['rhok'] ) # [N x nfreq x npairs]
            output['sigk'] = np.squeeze( self.nmos_iterations['sigk'] ) # [N x nfreq x npairs]
            output['Ck'] = np.squeeze( self.nmos_iterations['Ck'] ) # [N x nfreq x npairs x npairs]
        
        self.nmos_iterations = {} # Empty the dictionary to save memory
        return output
    
    
    def compute_rho_sig(self, X, Z, phihat):
        """Compute the rho_ab, sigma_ab correlation and uncertainties

        For Internal Use of the OptimalStatistic. Users are not recommended to use this!

        This method calculates the rho_ab and sigma_ab for each pulsar pair using
        the OS' X, Z and phihat matrix products. Check Appendix A from 
        Pol et al. 2023, or Gersbach et al. 2024 for matrix product definitions.

        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phihat (numpy.ndarray): A vector of the diagonal unit-amplitude phi matrix.

        Returns:
            numpy.ndarray, numpy.ndarray: The rho_ab and sigma_ab pairwise correlations
        
        Raises:
            ValueError: If NaN values are found in the pair-wise correlations.
        """
        # rho_ab = (X[a] @ phihat @ X[b]) / tr(Z[a] @ phihat @ Z[b] @ phihat)
        # sig_ab = np.sqrt( tr(Z[a] @ phihat @ Z[b] @ phihat) )
        a,b = self.pair_idx[:,0], self.pair_idx[:,1]

        numerator = np.einsum('ij,ij->i',X[a],(phihat*X[b]))
        Zphi = phihat*Z
        denominator = np.einsum('ijk,ikj->i',Zphi[a],Zphi[b])

        rho_ab = numerator/denominator
        sig_ab = 1/np.sqrt(denominator)

        if np.isnan(sig_ab).any():
            print(os_ex.NaNPairwiseError.extended_response())
            raise ValueError('NaN values pair-wise uncertainties! Are params valid?')

        return rho_ab, sig_ab


    def compute_rhok_sigk(self, X, Z, phi, freq, narrowband):
        """Compute the rho_ab(f_k), sigma_ab(f_k), and normalization_ab(f_k)

        For Internal Use of the OptimalStatistic. Users are not recommended to use this!

        This method calculates the rho_ab(f_k), sigma_ab(f_k), normalization_ab(f_k) 
        for each pulsar pair at the frequency index `freq`. Details of this implementation
        can be found in Gersbach et al. 2024. Note that the shape of the returned
        matrices are [n_pair].

        You can select to use the narrowband-normalization instead of the default
        broadband-normalization by setting the 'narrowband' argument to True.

        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phi (numpy.ndarray): A vector of the diagonal phi matrix.
            freq (int): The index of the frequency to compute the pair values at.
            narrowband (bool): Whether to use the narrowband-normalization instead
                    of the default broadband-normalization.

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray): rho_ab(f_k), sigma_ab(f_k), 
                                                         normalization_ab(f_k)
        """

        # Compute rho_ab(f_{freq}), sigma_ab(f_{freq}), and normalization_ab(f_{freq})
        a,b = self.pair_idx[:,0], self.pair_idx[:,1]

        phi_til = np.zeros(self.nfreq)  
        phi_til[freq] = 1
        phi_til = np.repeat(phi_til,2)
        phi2 = phi/phi[2*freq]

        if narrowband:
            norms_abk = 1/(np.einsum('ijk,ikj->i',phi_til*Z[a],phi_til*Z[b]))
        else:
            norms_abk = 1/(np.einsum('ijk,ikj->i',phi_til*Z[a],phi2*Z[b]))

        rho_abk =  np.sum(X[a] * phi_til * X[b], axis=1) * norms_abk
        sig_abk =  np.sqrt(np.einsum('ijk,ikj->i', phi_til*Z[a], phi_til*Z[b]) * norms_abk**2)

        if np.isnan(sig_abk).any():
            print(os_ex.NaNPairwiseError.extended_response())
            raise os_ex.NaNPairwiseError('NaN values pair-wise uncertainties! Are params valid?')

        return rho_abk, sig_abk, norms_abk


    # Handy matrix products-----------------------------------------------------
    def compute_XZ(self, params):
        """A method to quickly calculate the OS' matrix quantities

        This method calculates the X and Z matrix quantities from the appendix A
        of Pol, Taylor, Romano, 2022: (https://arxiv.org/abs/2206.09936). X and Z
        can be represented as X = F^T @ P^{-1} @ r and Z = F^T @ P^{-1} @ F.

        This method will change how P^{-1} is calculated depending on if you are
        using a linearized timing model (i.e. replace all T with F).
        
        Args:
            params (dict): A dictionary containing the parameter name:value pairs for the PTA

        Returns:
            (np.array, np.array): A tuple of X and Z. X is an array of vectors for each pulsar 
                (N_pulsar x 2N_frequency). Z is an array of matrices for each pulsar 
                (N_pulsar x 2N_frequency x 2N_frequency)
        """
        X = np.zeros( shape = ( self.npsr, 2*self.nfreq ) ) # An array of vectors
        Z = np.zeros( shape = ( self.npsr, 2*self.nfreq, 2*self.nfreq ) ) # An array of matrices

        for i in range(self.npsr):
            FNdt = self._get_FNdt(i, params)
            TNdt = self._get_TNdt(i, params)
            FNF = self._get_FNF(i, params)
            FNT = self._get_FNT(i, params)
            TNT = self._get_TNT(i, params)

            sigma = self._get_phiinv(i, params) + TNT

            X[i] = FNdt - FNT @ np.linalg.solve(sigma, TNdt)
            Z[i] = FNF - FNT @ np.linalg.solve(sigma, FNT.T)

        return X, Z


    # Matrix elements-----------------------------------------------------------
    # Don't cache these, they can have large memory footprints!
    def _get_F_mat(self, idx, params):
        return self.psr_signals[idx][self.gwb_name].get_basis(params)


    def _get_T_mat(self, idx, params):
        return self.psr_signals[idx].get_basis(params)


    def _get_N_obj(self, idx, params):
        return self.psr_signals[idx].get_ndiag(params)


    def _get_dt(self, idx, params):
        return self.psr_signals[idx].get_detres(params)


    def _get_phiinv(self, idx, params):
        phiinv = self.psr_signals[idx].get_phiinv(params)
        if phiinv.ndim == 1:
            phiinv = np.diag(phiinv)
        return phiinv 
    
    
    # Compute x^T N y-----------------------------------------------------------
    def _compute_xNy(self, N, x, y):
        if self.marginalizing_timing_model:
            # The marginalizing timing model doesn't support 
            # different left and right arrays :(
            term1 = N.Nmat.solve(y, left_array=x)
            term2 = N.MNF(x).T @ N.MNMMNF(y)
            return term1 - term2
        else:
            return N.solve(y, x)
    

    # Matrix products-----------------------------------------------------------
    @product_cache(['white_noise', 'basis', 'delay'], save_stats=True)
    def _get_FNdt(self, idx, params):
        F = self._get_F_mat(idx, params)
        N = self._get_N_obj(idx, params)
        dt = self._get_dt(idx, params)
        return self._compute_xNy(N, F, dt)


    @product_cache(['white_noise', 'basis', 'delay'], save_stats=True)
    def _get_TNdt(self, idx, params):
        T = self._get_T_mat(idx, params)
        N = self._get_N_obj(idx, params)
        dt = self._get_dt(idx, params)
        return self._compute_xNy(N, T, dt)


    @product_cache(['white_noise', 'basis'], save_stats=True)
    def _get_FNF(self, idx, params):
        F = self._get_F_mat(idx, params)
        N = self._get_N_obj(idx, params)
        return self._compute_xNy(N, F, F)


    @product_cache(['white_noise', 'basis'], save_stats=True)
    def _get_TNT(self, idx, params):
        T = self._get_T_mat(idx, params)
        N = self._get_N_obj(idx, params)
        return self._compute_xNy(N, T, T)


    @product_cache(['white_noise', 'basis'], save_stats=True)
    def _get_FNT(self, idx, params):
        F = self._get_F_mat(idx, params)
        N = self._get_N_obj(idx, params)
        T = self._get_T_mat(idx, params)
        return self._compute_xNy(N, F, T)

