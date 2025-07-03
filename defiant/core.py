
from . import custom_exceptions as os_ex
from . import utils
from . import pair_covariance as pc 
from . import null_distribution as os_nu
from . import orf_functions as orf_funcs

import numpy as np
import scipy.linalg as sl

import enterprise.signals as ent_sig
from enterprise.pulsar import BasePulsar

from la_forge.core import Core

from tqdm.auto import tqdm
from warnings import warn


class OptimalStatistic:
    """A class to compute the various forms of the Optimal Statistic for a given PTA.

    This class is designed in such a way to be able to combine all of the various
    generalizations of the Optimal Statistic into a single, cohesive class. This class
    can be made to compute any of the froms shown in the defiant choice tree in the 
    documentation of defiant. 
    NOTE: This class does not currently support PTAs with varied white noise parameters

    Attributes:
        psrs (list): A list of enterprise.pulsar.BasePulsar objects.
        pta (enterprise.signals.signal_base.PTA): A PTA object.
        npsr (int): The number of pulsars in the PTA.
        gwb_name (str): The name of the GWB in the PTA object.
        lfcore (la_forge.core.Core): A la_forge.core.Core object.
        max_like_params (dict): The maximum likelihood parameters from the chain.
        freqs (np.ndarray): The frequencies of the PTA.
        nfreq (int): The number of frequencies in the PTA.
        pair_names (list): A list of the used pulsar pairs.
        npair (int): The number of pulsar pairs.
        norfs (int): The number of overlap reduction functions.
        orf_functions (list): A list of the ORF functions.
        orf_design_matrix (np.ndarray): The design matrix of the ORFs.
        orf_names (list): The names of the ORFs.
        nside (int): The nside of the pixel basis (if using anisotropy).
        lmax (int): The maximum l value of the spherical harmonics (if using anisotropy).
        nmos_iterations (dict): A dictionary of the NMOS iterations.
        sub_tqdm (bool): Whether to use progress bars on frequency and pair covariance calculations.
        _pair_idx (np.ndarray): The index of the pulsar pairs.
        _orf_matrix (np.ndarray): A pulsar matrix of ORF values.
        _mcos_orf (np.ndarray): The ORF for the pair covariance matrix if using 
            the experimental MCOS.
        _max_chunk (int): The maximum number of simultaneous matrix products.
        _marginalizing_timing_model (bool): Whether the PTA is marginalizing the timing model.
        _cache (dict): A dictionary of cached matrix products
    """

    def __init__(self, psrs, pta, gwb_name='gw', core_path=None, core=None,  
                 chain_path=None, chain=None, param_names=None, 
                 orfs=['hd'], orf_names=None, pcmc_orf=None, clip_z=None,
                 sub_tqdm=False):
        """Initializes the OptimalStatistic object.

        There are many ways to initialize the OptimalStatistic object, and most
        parameters are optional. The most basic way to initialize the object is to
        call this initializer with a list of pulsars, a PTA object, and a gwb_name. 
        You may also need to use the set_chain_params() and set_orf() functions to
        set the MCMC chains and ORFs respectively. For convienence, the parameters
        for these functions are also available here in the initializer.
        NOTE: This class does not currently support PTAs with varied white noise parameters

        For info on the corepath, core, chain_path, chain, and params_names check
        documentation of OptimalStatistic.set_chain_params()
        
        For info on the orfs, and orf_names check documentation of OptimalStatistic.set_orf()

        *Experimental*
        The clip_z parameter is an experimental feature that can be used to set the
        minimum eigenvalue of the Z matrix products. This can be useful when the data
        is very noisy and leads to non-positive-definite matrices. 
        The clip_z parameter represents the minimum allowed eigenvalue of the Z matrix
        when normalized so that the maximum eigenvalue is 1.0. In this way, you cap
        the maximum condition number for all Z products to 1/clip_z. This generally 
        should be kept near machine precision. If it is needed, try setting this to
        a small value i.e. 1e-16 first before increasing it.
        Setting this value to None will disable the clipping, and should always be
        the default option unless you are experiencing issues like NaNPairwiseError.

        Args:
            psrs (list): A list of enterprise.pulsar.BasePulsar objects.
            pta (enterprise.signals.signal_base.PTA): A PTA object.
            gwb_name (str): The name of the GWB in the PTA object. Defaults to 'gw'.
            corepath (str, optional): The location of a pre-saved Core object.
            core (la_forge.core.Core, optional): A la_forge.core.Core object.
            chain_path (str, optional): The location of an PTMCMC chain.
            chain (np.ndarray, optional): The sampled chain.
            param_names (str or list, optional): The names of the chain parameters.
            orfs (list, optional): An orf name or function or list of orf names 
                or functions. See OptimalStatistic.set_orf(). Defaults to ['hd'].
            orf_names (str or list, optional): The names of the corresponding orfs. 
                See OptimalStatistic.set_orf(). Set to None for default names.
            pcmc_orf (str or function, optional): The assumed ORF for the pair 
                covariance matrix when using the MCOS. See OptimalStatistic.set_orf(). 
                Defaults to None.
            max_chunk (int, optional): The number of allowed simultaneous matrix 
                products to compute. Defaults to 300.
            clip_z (float, optional): (Experimental) The minimum eigenvalue of the 
                Z matrix products. Can be useful with very noisy data. Set to None 
                for no clipping. See doc comments for details. Defaults to None.
            sub_tqdm (bool, optional): Whether to use progress bars on frequency and
                pair covariance calculations. Defaults to False.

        Raises:
            TypeError: If the PTA object is not of type 'enterprise.signals.signal_base.PTA'.
            TypeError: If the pulsars in the psrs list are not a list or of type 
                'enterprise.pulsar.BasePulsar'.
        """
        self.sub_tqdm = sub_tqdm # Additional progress bars if needed

        if type(pta) == ent_sig.signal_base.PTA:
            self.pta = pta
        else:
            raise TypeError("pta supplied is not of type 'enterprise.signals.signal_base.PTA'!")
        try: 
            _ = psrs[0] 
        except TypeError: 
            raise TypeError("psrs list supplied is not able to be indexed")
        # Check for marginalizing timing model
        self._marginalizing_timing_model = False
        for s in self.pta._signalcollections[0].signals:
            if 'marginalizing linear timing model' in s.signal_name:
                self._marginalizing_timing_model = True

        # Check if this PTA has deterministic signals
        self.deterministic_signals = _check_deterministic(self.pta)
        
        self.gwb_name = gwb_name

        # Pre-cache matrix quantities
        self._cache = {}
        self._compute_cached_matrices()


        # Duck typing pulsars
        self.psrs = psrs
        self.npsr = len(psrs)
        psr_names = self.pta.pulsars
        
        
        self.lfcore, self.max_like_params, self.max_like_idx = None, None, None
        self.set_chain_params(core, core_path, chain_path, chain, param_names)

        self.freqs = utils.get_pta_frequencies(pta,gwb_name)
        self.nfreq = len(self.freqs) 
        
        self._pair_idx = np.array(np.triu_indices(self.npsr,1)).T
        self.pair_names = [(psr_names[a],psr_names[b]) for a,b in self._pair_idx]
        self.npairs = len(self.pair_names)

        self.norfs, self.orf_functions = 0, []
        self.orf_design_matrix, self._orf_matrix, self._mcos_orf = None, None, None
        self.orf_names = None
        self.set_orf(orfs, orf_names, pcmc_orf)
        self.nside, self.lmax = None, None 

        self.nmos_iterations = {} # Used to store the NMOS iterations
        self._max_chunk = 300 # Users can set after creation if they want to change it

        # Experimental stuff ---------------------------------------------------

        self.clip_z = clip_z
        if clip_z is not None:
            warn("Clipping Z matrix products is an experimental feature. Use with caution.")
        

    def set_chain_params(self, core=None, core_path=None, chain_path=None, 
                         chain=None, param_names=None):
        """A method to add MCMC chains to an OptimalStatistic object. 

        This method takes a number of different forms of MCMC chains and creates
        a la_forge.core.Core object for use with noise marginalization or maximum 
        likelihood optimal statistic. To use this function, you must include one
        or more of the following options from most prefered to least:
            1. core 
            2. corepath   
            3. chain_path  
            4. chain & param_names   
            5. chain

        The core object can then be accessed through self.lfcore with the maximum
        likelihood parameters stored in self.max_like_params or through its chain
        index in self.max_like_idx.

        Args:
            core (la_forge.core.Core, optional): A Core object. 
            corepath (str, optional): The path to a saved la_forge.core.Core object. 
            chain_path (str, optional): The path to a saved chain from PTMCMC. 
            chain (numpy.ndarray, optional): The raw numpy.ndarray of the MCMC chain. 
            param_names (list, optional): The order of the parameter names of chain. 
        """
        # Prefered order for loading chains: 
        # core > corepath > chain_path > chain + param_names > chain
        if core is not None and type(core) == Core:
            self.lfcore = core
        elif core_path is not None:
            self.lfcore = Core(corepath=core_path)
        elif chain_path is not None:
            self.lfcore = Core(chaindir=chain_path)
        elif chain is not None and param_names is not None:
            self.lfcore = Core(chain=chain,params=param_names)
        elif chain is not None:
            self.lfcore = Core(chain=chain,params=self.pta.param_names)
        else:
            msg = 'No MCMC samples were given! Set these later or supply ' +\
                  'them when computing the OS.'
            #warn(msg)

        if self.lfcore is not None:
            params, idx = utils.get_max_like_params(self.lfcore)
            self.max_like_params = params
            self.max_like_idx = idx


    def get_chain_params(self, N=1, idx=None, format='dict', freespec_fix=False, 
                         return_idx=False):
        """A method to get samples from the self.lfcore object.

        This method is a helper method to either get random samples from the chain
        or to get specific samples if 'idx' is supplied. This function can be useful
        when you want to know which parameters were used in a particular iteration
        of the noise marginalizing process. This can especially be helpful with 
        varried gamma CURN models.

        If 'idx' is None then this method will return N random samples from the chain.
        If 'idx' is either an int or an array of ints, then this method will return
        the samples at those specific indexes. 
        NOTE: 'idx' must refer to the unburned indexes of the chain. (i.e. idx=0 will
        be the first sample in the chain, which will usually be in the burn-in.)
        NOTE: N is ignored if idx is supplied.

        The 'format' parameter can be set to either 'dict' or 'array'. If 'dict' is
        chosen, then the samples will be returned as a list of dictionaries where the
        keys are the parameter names. If 'array' is chosen, then the samples will be
        returned as a numpy.ndarray.
        NOTE: If using 'array' to get the parameters associated with each 
        index i in array[:,i] you can use self.lfcore.params.

        The 'freespec_fix' parameter is used to fix a minor bug in enterprise which
        expects the freespec GWB parameters to be of the form 'gw_log10_rho' instead
        of the sampled 'gw_log10_rho_0', 'gw_log10_rho_1', .... Enabling this
        flag will add an additional key value pair to the dictionary return. 
        NOTE: This will only be applied to the 'dict' format.

        The 'return_idx' parameter is used to return the indexes of the samples that
        were returned. This can be useful if you wish to know which index the samples
        were taken from. 

        Args:
            N (int, optional): The number of random samples to generate. Defaults to 1.
            idx (int or np.ndarray, optional): The index or indexes of the samples to return.
                Set to None for random samples. Defaults to None.
            format (str): The format of the return. Can be 'dict' or 'array'.
                Defaults to 'dict'.
            freespec_fix (bool): Whether to fix the freespec GWB parameter names 
                to make Enterprise happy. Defaults to False.
            return_idx (bool): Whether to return the indexes of the samples. Defaults to False.

        Raises:
            ValueError: If the chain parameters are not set. Set with self.set_chain_params().
            ValueError: If the format is not either 'dict' or 'array'.

        Returns:
            The outputs of this method change depending on the 'format' and 'return_idx'

            If 'return_idx' is False:
                - Either a list of dictionaries or a numpy.ndarray of the samples.
            If 'return_idx' is True:
                - A tuple where the first element is the samples and the second is the indexes.
        """

        if self.lfcore is None:
            msg = 'No chain parameters set! Set these before calling this function.'
            raise ValueError(msg)
        
        if idx is None:
            # If indexes are not supplied, then choose them randomly without burn-in
            idx = np.random.randint(self.lfcore.burn, self.lfcore.chain.shape[0], N)
        else:
            if type(idx) == int:
                idx = [idx]
            idx = np.array(idx)

        
        if format == 'array':
            if return_idx:
                return self.lfcore.chain[idx], idx
            return self.lfcore.chain[idx]
        elif format == 'dict':
            ret = []
            for i in idx:
                d = {p:v for p,v in zip(self.lfcore.params, self.lfcore.chain[i])}
                if not freespec_fix:
                    ret.append(d)
                else:
                    # This changes the 'gw_log10_rho_0', 'gw_log10_rho_1', ... into
                    # a single 'gw_log10_rho'
                    nd = utils.freespec_param_fix(d, self.gwb_name)
                    ret.append(nd)
            if return_idx:
                return ret, idx
            return ret
        else:
            msg = f"Format {format} not recognized. Use 'dict' or 'array'."
            raise ValueError(msg)


    def set_orf(self, orfs=['hd'], orf_names=None, pcmc_orf=None):
        """Sets the overlap reduction function[s] (ORF) for the cross correlations.

        Sets the overlap reduction function[s] of the cross correlation and the
        corresponding ORF design matrix. This function supports multiple ORFs by
        simply supplying a list of ORFs. orf_names can be left as None to use default
        names. orfs can also be a user-defined function which accepts 2 
        enterprise.pulsar.BasePulsar. 

        Otherwise, use one of the following pre-defined ORF within 
        defiant.orf_functions.defined_orfs:
            - 'hd' or 'hellingsdowns': Hellings and Downs
            - 'dp' or 'dipole': Dipole
            - 'mp' or 'monopole': Monopole
            - 'gwdp' or 'gw_dipole': Gravitational wave dipole
            - 'gwmp' or 'gw_monopole': Gravitational wave monopole
            - 'st' or 'scalar_tensor': Scalar tensor
            - 'l_' or 'legendre_': Legendre polynomials where the number after 
                    the _ is the degree
            
        *Experimental*
        pcmc_orf is an experimental feature that aims to curb the problematic 
        nature of pair covariance matrix with the MCOS. In order to construct
        the pair covariance matrix, an assumed ORF must be used. With a single 
        ORF, the answer is trivial, but with multiple ORFs, the ratio of power
        for each component is not known a priori. This can lead to large changes
        in the covariance matrix for minor changes in the assumed power in each
        component. The pcmc_orf parameter allows the user to set an assumed ORF
        which will override the default behavior of the MCOS. This can be especially
        usefull when using many ORFs or components of the ORF are expected to be
        near zero.

        If this argument is set to None (default): 
            - The typical behavior will be used, this being that the power per 
            process will be the normalized non-pair covariant MCOS multiplied 
            by the CURN amplitude. 
        If this argument is set to a str: 
            - The pair covariance matrix will be computed using a singular ORF 
            function specified (i.e. 'hd').
        If this argument is set to a function:
            - The pair covariance matrix will be computed using the user supplied 
            function. This function must accept two enterprise.pulsar.BasePulsar 
            objects as inputs and outputs a float for their ORF.

        
        Args:
            orfs (str or function or list): An ORF string, function or list of 
                strings and/or functions. Note that a custom function must 
                accept two enterprise.pulsar.BasePulsar objects as inputs 
                and outputs a float for their ORF.
            orf_names (list, optional): The names of the corresponding orfs. Set to None
                for default names.
            pcmc_orf (str or function, optional): The assumed ORF for the pair covariance 
                    matrix. when using the MCOS. Defaults to None.

        Raises:
            ValueError: If the length of the orfs and orf_names does not match.
            ORFNotFoundError: If a pre-defined ORF is not found.
            TypeError: If the user-supplied ORF does not have correct format. 
        """
        if not hasattr(orfs, '__iter__'):
            orfs = [orfs]
        elif type(orfs) == str:
            orfs = [orfs]
        
        if orf_names is None:
            orf_names = [None for a in orfs]
        elif type(orfs) == str:
            orf_names = [orf_names]

        # Check for same length!
        if len(orfs) != len(orf_names):
            msg = 'length of orfs and length of orf_names is not equal!'
            raise ValueError(msg)
        
        self.norfs = len(orfs)
        self.orf_names = []
        # Used internal and external
        self.orf_design_matrix = np.zeros( (self.norfs,self.npairs) ) 
        # Used internal for pair covariance
        self._orf_matrix = np.ones( (self.norfs,self.npsr,self.npsr) ) 
        
        self.orf_functions = []

        for i in range( len(orfs) ):
            orf = orfs[i]
            name = orf_names[i]

            if type(orf) == str:
                # ORF must be pre-defined function
                cur_orf = orf_funcs.get_orf_function(orf)
                name = orf if name is None else name
                
                self.orf_names.append(name)
                self.orf_functions.append(cur_orf)
                for j,(a,b) in enumerate(self._pair_idx):
                    v = cur_orf(self.psrs[a] , self.psrs[b])
                    self.orf_design_matrix[i,j] = v
                    self._orf_matrix[i,a,b] = v
                    self._orf_matrix[i,b,a] = v

            else:
                # ORF is user supplied function
                name = orf.__name__ if name is None else name

                self.orf_names.append(name)
                self.orf_functions.append(orf)
                try:
                    for j,(a,b) in enumerate(self._pair_idx):
                        v = orf(self.psrs[a], self.psrs[b])
                        self.orf_design_matrix[i,j] = v
                        self._orf_matrix[i,a,b] = v
                        self._orf_matrix[i,b,a] = v
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
            if type(pcmc_orf) == str:
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
            self._mcos_orf = temp
        else:
            self._mcos_orf = None


    def set_anisotropy_basis(self, basis='pixel', nside=2, lmax=6, pc_orf='hd'):
        """A method to set the anisotropy basis for the OS.

        This function sets the anisotropy basis for the OS. The basis can be either
        a pixel basis or a spherical harmonic basis. The pixel basis is a simple
        pixelization of the sky, while the spherical harmonic basis is a spherical
        harmonic decomposition of the sky. If the basis is set to 'pixel', lmax is
        ignored. For pair covariance to work correctly, an assumed ORF must be set,
        which is done with the pc_orf argument. This must be a pre-defined ORF found
        in the orf_functions.defined_orfs list.

        Args:
            basis (str): The basis for the anisotropy. Must be 'pixel' or 'spherical'.
            nside (int): The nside of the pixelization. Defaults to 2.
            lmax (int, optional): The maximum l value of the spherical harmonics. Defaults to 6.
            pc_orf (str): The ORF to use for the pair covariance matrix. Defaults to 'hd'.
        """
        orf = orf_funcs.get_orf_function(pc_orf)
        temp = np.zeros( (self.npsr,self.npsr) )
        for a in range(self.npsr):
            for b in range(a+1,self.npsr):
                v = orf(self.psrs[a],self.psrs[b])
                temp[a,b] = v
                temp[b,a] = v
        self._mcos_orf = temp

        if basis.lower() == 'pixel':
            basis = orf_funcs.anisotropic_pixel_basis(self.psrs, nside, self._pair_idx)
            self.nside = nside
            self.lmax = None

            self.orf_names = [f'pixel_{i}' for i in range(basis.shape[1])]
            self.orf_design_matrix = basis
            self._orf_matrix = None
            self.norfs = basis.shape[1]

        elif basis.lower() == 'spherical':
            basis = orf_funcs.anisotropic_spherical_harmonic_basis(self.psrs, lmax, 
                                                                nside, self._pair_idx)
            self.nside = nside
            self.lmax = lmax

            self.orf_names = [f'c_{l},{m}' for l in range(lmax+1) for m in range(-l,l+1)]
            self.orf_design_matrix = basis
            self._orf_matrix = None
            self.norfs = basis.shape[1]
    
        else:
            raise os_ex.ORFNotFoundError(f"Anisotropy basis {basis} not found!")
        

    def compute_OS(self, params=None, N=1, gamma=None, pair_covariance=False, 
                   return_pair_vals=True, fisher_diag=False, use_tqdm=True):
        """Compute the OS and its various modifications.

        This is one of 2 main functions of the OptimalStatistic class. This function
        can compute any flavor of the OS which uses broadband estimation (i.e. constructs
        a single estimator for the whole spectrum). There are many forms in which you can
        use this function, and checking the decision tree is best for determining exactly
        what you might want and what parameters to set to accomplish that. 
        
        The basic usage of this function can be boiled down to the following:
        If you want to compute a single iteration of the OS:
            - supply a set of params and set N=1. By default, if params=None, this 
              will compute the maximum likelihood OS.
        If you want to compute the noise marginalized OS:
            - ensure that the OptimalStatistiic object has a La forge core set 
              (see OptimalStatistic.set_chain_params), and set N>1. 
        If you want to compute the OS with pair covariance:
            - simply set pair_covariance=True. This will also replace the covariance 
              matrix, C, that gets returned.
        If you are using a varied gamma CURN model, you can either:
            - Set a particular gamma value for all NMOS iterations by setting gamma
            - Or set gamma=None and the function will default to each iterations' gamma value
            or the fixed gamma value if used

        Users can choose to forgo returning the pairwise values (i.e. rho, sig, C)
        when executing this method by setting return_pair_vals=False. This can be useful
        when you are only interested in the OS estimators and wish to save memory as
        the pairwise covariance matrices can be very large. 

        The 'params' argument also has the additional functionality to allow users
        to supply specific parameter index or indexes by setting params to an int or
        a list of ints.

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component OS processes
        simultaneously. Keep this on True unless you know what you are doing.

        Args:
            params (dict, list, int, optional): The parameters to use in the OS. 
                Users can supply any of the following for this argument:
                    - dict: A dictionary of parameter values
                    - list: A list of dictionaries of parameter values
                    - int: An index of a parameter vector in the chain
                    - list: A list of indexes of parameter vectors in the chain
                    - None: If N==1, defaults to maximum likelihood.
                    - None: If N>1, use random samples from the chain. (NMOS)
            N (int): The number of NMOS iterations to run. If params is not None, 
                this is argument ignored.
            gamma (float, optional): The spectral index to use for analysis. If set to None,
                this function first checks if gamma is in params, otherwise it will
                assume the PTA model is a fixed gamma and take it from there. Defaults to None.
            pair_covariance (bool): Whether to use pair covariance. Defaults to False.
            return_pair_vals (bool): Whether to return the xi, rho, sig, C values. Defaults to True.
            fisher_diag (bool): Whether to zero the off-diagonal elements of the
                fisher matrix. Defaults to False.
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.

        Returns:
            tuple: Return values are very different depending on which options you choose.
            Every value is either returned as an np.array or float depending on context.

            If return_pair_vals is True:
                - returns xi, rho, sig, C, A2, A2s, param_index
            If return_pair_vals is False:
                - returns A2, A2s, param_index
            
            xi (np.ndarray) - The pair separations of the pulsars [npairs]
            rho (np.ndarray) - The pair correlated powers [N x npairs]
            sig (np.ndarray) - The pair uncertainties in rho [N x npairs]
            C (np.ndarray) - The pair covariance matrix [N x npairs x npairs]
            A2 (np.ndarray/float) - The OS amplitude estimators at 1/yr [N x Norfs]
            A2s (np.ndarray/float) - Can be either:
                - The 1-sigma uncertainties of A2 if norfs==1 [N]
                - A covariance matrix on the fits if norfs>1 [N x Norfs x Norfs]
            param_index (np.ndarray) - The index(es) of the parameter vectors used 
                in NM each iteration. If no indexes are found indexes will be -1 [N]
        """
        # First, get the parameter dictionaries
        all_pars, all_idx, all_gamma = self._parse_params(params, N, gamma)    

        # Need GWB signal to compute phi
        gw_signal = [s for s in self.pta._signalcollections[0].signals if s.signal_id==self.gwb_name][0]

        # Now setup the return values inside of the self.nmos_iterations
        self.nmos_iterations = {'A2':[],'A2s':[],'param_index':[]}
        if return_pair_vals:
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rho'] = []
            self.nmos_iterations['sig'] = []
            self.nmos_iterations['C'] = []   

        if use_tqdm and len(all_pars)>1:
            iterable = tqdm(range(len(all_pars)),desc='NMOS Iters')
        else:
            # Single iteration or no tqdm
            iterable = range(len(all_pars))

        for i in iterable:
            par, idx, gam = all_pars[i], all_idx[i], all_gamma[i]
            phihat = utils.powerlaw(self.freqs, 0, gam, 2) # Sine and Cosine

            # Get the handy OS matrix products
            X, Z = self._compute_XZ(par)

            # Now compute the pairwise correlations and uncertainties
            rho_ab, sig_ab = self._compute_rho_sig(X, Z, phihat)

            # Compute the covariance matrix
            if pair_covariance:
                method = 'woodbury'
                a2_est = utils.get_a2_estimate(par, self.freqs, gam, self.gwb_name, self.lfcore)

                # Compute the pair covariance matrix: 3 behaviors
                if self.norfs==1:
                    # Single ORF (phi = A2*phihat)
                    C = pc.create_OS_pair_covariance(Z, phihat, a2_est*phihat, self._orf_matrix[0], 
                            sig_ab**2, use_tqdm and self.sub_tqdm, self._max_chunk)
                        
                elif self.norfs>1 and self._mcos_orf is not None:
                    # Assumed ORF with MCOS
                    C = pc.create_OS_pair_covariance(Z, phihat, a2_est*phihat, self._mcos_orf, 
                            sig_ab**2, use_tqdm and self.sub_tqdm, self._max_chunk)
                        
                else:
                    # Default behavior
                    C = pc.create_MCOS_pair_covariance(Z, phihat, self._orf_matrix, 
                            self.orf_design_matrix, rho_ab, sig_ab, a2_est, 
                            use_tqdm and self.sub_tqdm, self._max_chunk)
                
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
            self.nmos_iterations['param_index'].append(idx)
            if return_pair_vals:
                self.nmos_iterations['rho'].append(rho_ab)
                self.nmos_iterations['sig'].append(sig_ab)
                self.nmos_iterations['C'].append(C)
        
        # Setup our return values
        A2 = np.squeeze( self.nmos_iterations['A2'] )
        A2 = A2.item() if A2.size==1 else A2
        A2s = np.squeeze( self.nmos_iterations['A2s'] )
        A2s = A2s.item() if A2s.size==1 else A2s
        param_index = np.squeeze( self.nmos_iterations['param_index'] )
        param_index = param_index.item() if param_index.size==1 else param_index

        if return_pair_vals:
            xi = np.squeeze( self.nmos_iterations['xi'] )
            rho = np.squeeze( self.nmos_iterations['rho'] )
            sig = np.squeeze( self.nmos_iterations['sig'] )
            C = np.squeeze( self.nmos_iterations['C'] )
            
            self.nmos_iterations = {}
            return xi, rho, sig, C, A2, A2s, param_index
        
        self.nmos_iterations = {}
        return A2, A2s, param_index


    def compute_PFOS(self, params=None, N=1, pair_covariance=False, narrowband=False, 
                     return_pair_vals=True, fisher_diag=False, select_freq=None, use_tqdm=True):
        """Compute the PFOS and its various modifications.

        This is one of 2 main functions of the OptimalStatistic class. This function
        can computes the different flavors of the PFOS (i.e. a free-spectrum search). 
        There are many forms in which you can use this function, and checking the 
        decision tree is best for determining exactly what you might want and what 
        parameters to set to accomplish that. 
        
        The basic usage of this function can be boiled down to the following:
        If you want to compute a single iteration of the PFOS:
            - supply a set of params and set N=1. By default, if params=None, this 
              will compute the maximum likelihood PFOS.
        If you want to compute the noise marginalized PFOS:
            - ensure that the OptimalStatistiic object has a La forge core set 
              (see OptimalStatistic.set_chain_params), and set N>1. 
        If you want to compute the PFOS with pair covariance:
            - simply set pair_covariance=True. This will also replace the covariance 
              matrix, C, that gets returned.

        The 'params' argument also has the additional functionality to allow users
        to supply specific parameter index or indexes by setting params to an int or
        a list of ints.

        Users can choose to forgo returning the pairwise values (i.e. rhok, sigk, Ck)
        when executing this method by setting return_pair_vals=False. This can be useful
        when you are only interested in the PFOS estimators and wish to save memory as
        the pairwise covariance matrices can be very large. 

        The 'narrowband' argument is used to compute the narrowband-normalized PFOS
        detailed in Gersbach et al. 2024. This makes an additional simplification 
        to the PFOS at the cost of accuracy in detectable GWB signals. This is mostly
        included for legacy and bugfixing purposes.

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component PFOS processes
        simultaneously. Keep this on True unless you know what you are doing.

        If you want to select a specific frequency to compute the PFOS at, you can set
        the 'select_freq' parameter to the index of the frequency you wish to compute the
        PFOS at. This can be useful if you only care about a specific frequency. Setting
        this to None will compute the PFOS at all frequencies.

        Args:
            params (dict, list, int, optional): The parameters to use in the PFOS. 
                Users can supply any of the following for this argument:
                    - dict: A dictionary of parameter values
                    - list: A list of dictionaries of parameter values
                    - int: An index of a parameter vector in the chain
                    - list: A list of indexes of parameter vectors in the chain
                    - None: If N==1, defaults to maximum likelihood.
                    - None: If N>1, use random samples from the chain. (NM PFOS)
            N (int): The number of NM PFOS iterations to run. If params is not None, 
                this is argument ignored.
            narrowband (bool): Whether to use the narrowband-normalized PFOS instead of
                the default broadband-normalized PFOS. Defaults to False.
            pair_covariance (bool): Whether to use pair covariance. Defaults to False.
            return_pair_vals (bool): Whether to return the xi, rho, sig, C values. Defaults to True.
            fisher_diag (bool): Whether to zero the off-diagonal elements of the
                fisher matrix. Defaults to False.
            select_freq (int, optional): The index of the frequency to compute the PFOS at.
                Defaults to None.
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.

        Returns:
            tuple: Return values are very different depending on which options you choose.
            Every value is either returned as an np.array or float depending on context.

            If return_pair_vals is True:
                - returns xi, rhok, sigk, Ck, Sk, Sks, param_index
            If return_pair_vals is False:
                - returns Sk, Sks, param_index
            
            xi (np.ndarray) - The pair separations of the pulsars [npairs]
            rhok (np.ndarray) - The pair correlated powers per frequency [N x nfreq x npairs]
            sigk (np.ndarray) - The pair uncertainties in rhok [N x nfreq x npairs]
            Ck (np.ndarray) - The pair covariance matrix per frequency [N x nfreq x npairs x npairs]
            Sk (np.ndarray) - The PFOS PSD/Tspan estimators at each frequency [N x nfreq x Norfs]
            Sks (np.ndarray) - Can be either:
                - The 1-sigma uncertainties of Sk if norfs==1 [N x nfreq]
                - A covariance matrix on the fits if norfs>1 [N x nfreq x Norfs x Norfs]
            param_index (np.ndarray) - The index(es) of the parameter vectors used 
                in NM each iteration. If no indexes are found indexes will be -1 [N]
        """
        # First, get the parameter dictionaries. Set gamma, but ignore it anyway
        all_pars, all_idx, _ = self._parse_params(params, N, 1.0)     

        # Now setup the return values inside of the self.nmos_iterations
        self.nmos_iterations = {'Sk':[],'Sks':[],'param_index':[]}
        if return_pair_vals:
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rhok'] = []
            self.nmos_iterations['sigk'] = []
            self.nmos_iterations['Ck'] = []  

        # Need GWB signal to compute phi
        gw_signal = [s for s in self.pta._signalcollections[0].signals if s.signal_id==self.gwb_name][0]

        if use_tqdm and len(all_pars)>1:
            iterable = tqdm(range(len(all_pars)),desc='NMOS Iters')
        else:
            # Single iteration or no tqdm
            iterable = range(len(all_pars))

        for i in iterable:
            par, idx = all_pars[i], all_idx[i]
            # Get the handy OS matrix products
            X,Z = self._compute_XZ(par)

            # Need the approximate shape of the GWB spectrum
            phi = gw_signal.get_phi(par)

            # Compute the pairwise correlations, uncertainties and normalizations
            # for each frequency
            rho_abk, sig_abk, norm_abk = self._compute_rhok_sigk(X, Z, phi, narrowband, 
                                                                 select_freq)
            
            if pair_covariance:
                method = 'woodbury'
                if self.norfs==1:
                    # Single ORF
                    Ck = pc.create_PFOS_pair_covariance(Z, phi, self._orf_matrix[0], 
                            norm_abk, narrowband, select_freq,
                            use_tqdm and self.sub_tqdm, self._max_chunk)
                
                elif self.norfs>1 and self._mcos_orf is not None:
                    # Assumed ORF with MCOS
                    Ck = pc.create_PFOS_pair_covariance(Z, phi, self._mcos_orf, 
                            norm_abk, narrowband, select_freq, 
                            use_tqdm and self.sub_tqdm, self._max_chunk)
                
                else:
                    # Default MCOS behavior
                    Ck = pc.create_MCPFOS_pair_covariance(Z, phi, self._orf_matrix, 
                            norm_abk, self.orf_design_matrix, rho_abk, sig_abk, 
                            narrowband, select_freq,
                            use_tqdm and self.sub_tqdm, self._max_chunk)

            else:
                method = 'diagonal'
                if select_freq is None:
                    Ck = np.array([np.diag(sig_abk[k]**2) for k in range(self.nfreq)])
                else:
                    Ck = np.diag(sig_abk**2)

            # Done with pair covariance, now compute the PFOS
            if select_freq is not None:
                s_diag = np.diag(sig_abk**2)
                Sk, Sks = utils.linear_solve(self.orf_design_matrix, Ck, rho_abk,
                                             s_diag, method, fisher_diag)
                
                Sk = np.squeeze(Sk) if self.norfs>1 else Sk.item()
                Sks = np.squeeze(Sks) if self.norfs>1 else np.sqrt(Sks.item())

            else:


                Sk = np.zeros( (self.nfreq,self.norfs) ) 
                Sks = np.zeros( (self.nfreq,self.norfs,self.norfs) )
                for k in range(self.nfreq):
                    s_diag = np.diag(sig_abk[k]**2)
                    sk, sksig = utils.linear_solve(self.orf_design_matrix, Ck[k], rho_abk[k], 
                                                   s_diag, method, fisher_diag)
                
                    Sk[k] = np.squeeze(sk) if self.norfs>1 else sk.item()
                    Sks[k] = np.squeeze(sksig) if self.norfs>1 else np.sqrt(sksig.item()) 
            
            self.nmos_iterations['Sk'].append(np.squeeze(Sk))
            self.nmos_iterations['Sks'].append(np.squeeze(Sks))
            self.nmos_iterations['param_index'].append(idx)
            if return_pair_vals:
                self.nmos_iterations['rhok'].append(rho_abk)
                self.nmos_iterations['sigk'].append(sig_abk)
                self.nmos_iterations['Ck'].append(Ck)

        
        Sk = np.squeeze( self.nmos_iterations['Sk'] )
        Sks = np.squeeze( self.nmos_iterations['Sks'] )
        param_index = np.squeeze( self.nmos_iterations['param_index'] )

        if return_pair_vals:
            xi = np.squeeze( self.nmos_iterations['xi'] )
            rhok = np.squeeze( self.nmos_iterations['rhok'] )
            sigk = np.squeeze( self.nmos_iterations['sigk'] )
            Ck = np.squeeze( self.nmos_iterations['Ck'] )

            self.nmos_iterations = {}
            return xi,rhok, sigk, Ck, Sk, Sks, param_index

        self.nmos_iterations = {}
        return Sk, Sks, param_index
    

    def _compute_cached_matrices(self):
        """A function to compute the constant valued matrices used in the OS.

        This function will calculate the following matrix product, which are constant
        for any parameter values given to the PTA.
        The stored matrix products are:
            - FNr [2N_gwb x 1]
            - FNF [2N_gwb x 2N_gwb] 
            - FNT [2N_gwb x 2N_model]
            - TNT [2N_model x 2N_model]
            - TNr [2N_model x 1]

        The matrix products containing r, use the raw residuals of the pulsar, that
        is, the residuals WITHOUT subtracting any deterministic signals.

        Note: If the PTA is using the marginalizing timing model T will be replaced
        with F_irn, where F is the fourier design matrix of GWB only, and F_irn is
        the fourier design matrix of the full red noise.
        """
        all_FNr = []
        all_FNF = []
        all_FNT = [] 
        all_TNT = []
        all_TNr = []

        if self.sub_tqdm:
            iterator = tqdm(range(len(self.pta._signalcollections)), desc='Precompute matrices')
        else:
            iterator = range(len(self.pta._signalcollections))
            
        for idx in iterator:
            psr_signal = self.pta._signalcollections[idx]
            r = psr_signal._residuals # Raw residuals, no deterministic signals
            N = psr_signal.get_ndiag()
            T = psr_signal.get_basis()
            F = _get_F_matrices(self.pta, self.gwb_name, idx)
            # Getting the F matrix is a bit weird and very enterprise version dependent
            
            if self._marginalizing_timing_model:
                # Need to use own solving method for N
                FNr = _solveD(N,r,F) # F^T @ N^{-1} @ r
                TNr = _solveD(N,r,T) # T^T @ N^{-1} @ r
                FNF = _solveD(N,F,F) # F^T @ N^{-1} @ F
                TNT = _solveD(N,T,T) # T^T @ N^{-1} @ T
                FNT = _solveD(N,T,F) # F^T @ N^{-1} @ T
            else:
                FNr = N.solve(r,F) # F^T @ N^{-1} @ r
                TNr = N.solve(r,T) # T^T @ N^{-1} @ r
                FNF = N.solve(F,F) # F^T @ N^{-1} @ F
                TNT = N.solve(T,T) # T^T @ N^{-1} @ T
                FNT = N.solve(T,F) # F^T @ N^{-1} @ T
                
            all_FNr.append(FNr)
            all_FNF.append(FNF)
            all_TNT.append(TNT)
            all_FNT.append(FNT)
            all_TNr.append(TNr)

        self._cache['FNr'] = all_FNr
        self._cache['FNF'] = all_FNF
        self._cache['FNT'] = all_FNT
        self._cache['TNT'] = all_TNT
        self._cache['TNr'] = all_TNr

    def _compute_delay_FNr_TNr(self, params, idx):
        
        # If the real resiudals are dt = r - r', then FN(dt) = FNr - FNr' 
        # and TN(dt) = TNr - TNr', where r' is the set of delays on the residuals. 
        # Check if this pulsar has a deterministic signal, if so, compute the delay
        if self.deterministic_signals[idx]:
            # Yes deterministic signal, compute the delay FNr
            psr_signal = self.pta._signalcollections[idx]

            r_prime = psr_signal.get_delay(params)
            N = psr_signal.get_ndiag()
            T = psr_signal.get_basis()
            F = _get_F_matrices(self.pta, self.gwb_name, idx)

            if self._marginalizing_timing_model:
                # Need to use own solving method for N
                FNr_prime = _solveD(N, r_prime, F)
                TNr_prime = _solveD(N, r_prime, T)
            else:
                FNr_prime = N.solve(r_prime, F)
                TNr_prime = N.solve(r_prime, T)

            FNdt = self._cache['FNr'][idx] - FNr_prime # F^T @ N^{-1} @ (r - r')
            TNdt = self._cache['TNr'][idx] - TNr_prime # T^T @ N^{-1} @ (r - r')
            return FNdt, TNdt
        else:
            # No deterministic signal, just use the cached values
            FNdt = self._cache['FNr'][idx] 
            TNdt = self._cache['TNr'][idx]
            return FNdt, TNdt
    

    def _compute_XZ(self, params):
        """A function to quickly calculate the OS' matrix quantities

        This function calculates the X and Z matrix quantities from the appendix A
        of Pol, Taylor, Romano, 2022: (https://arxiv.org/abs/2206.09936). X and Z
        can be represented as X = F^T @ P^{-1} @ r and Z = F^T @ P^{-1} @ F.

        This function will change how P^{-1} is calculated depending on if you are 
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

        for a,psr_signal in enumerate(self.pta._signalcollections):
            phiinv = psr_signal.get_phiinv(params)
            if phiinv.ndim == 1:
                phiinv = np.diag(phiinv)

            # FNr and TNr can be modulated by deterministic signals
            FNr, TNr = self._compute_delay_FNr_TNr(params, a)

            FNF = self._cache['FNF'][a]
            FNT = self._cache['FNT'][a]
            TNT = self._cache['TNT'][a]
            

            sigma = phiinv + TNT
            # Previously did cholesky, but forward modeling is faster and more stable
            X[a] = FNr - FNT @ np.linalg.solve(sigma, TNr)
            Z[a] = FNF - FNT @ np.linalg.solve(sigma, FNT.T)

        if self.clip_z is not None:
            for i in range(len(Z)):
                Z[i] = utils.clip_covariance(Z[i], self.clip_z)

        return X, Z
    

    def _parse_params(self, params, N, gamma):
        """A helper method to parse the parameters for the OS and PFOS.

        This method is used to parse the parameters for the OS and PFOS. It is a
        helper method to the compute_OS() and compute_PFOS() methods. This method
        is used to normalize the parameters used in these methods while still allowing
        the user to use a variety of different usage formats.

        This method is hidden from the user and is not meant to be called directly.

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
            if type(params) is dict:
                # A single dictionary of parameters
                idx = [-1]
                pars = [utils.freespec_param_fix(params, self.gwb_name)]
            elif hasattr(params, '__iter__') and type(params[0]) is dict:
                # An iterable of dictionaries
                idx = [-1]*len(params)
                pars = [utils.freespec_param_fix(p, self.gwb_name) for p in params]
            elif hasattr(params, '__iter__') and type(params[0]) is int:
                # An iterable of indexes
                idx = params
                pars = self.get_chain_params(idx=params, format='dict', freespec_fix=True)
            elif type(params) is int:
                # A single index
                idx = [params]
                pars = [self.get_chain_params(idx=params, format='dict', freespec_fix=True)]
            else:
                msg = "params must be a dictionary, list of dictionaries, or list of indexes."
                raise ValueError(msg)
        elif N==1:
            # Default to maximum likelihood
            idx = [self.max_like_idx]
            pars = [utils.freespec_param_fix(self.max_like_params, self.gwb_name)]
        elif N>1:
            # Default to noise marginalization
            pars, idx = self.get_chain_params(N, format='dict', return_idx=True,
                                              freespec_fix=True)
        else:
            msg = "Unable to determine parameters. Set either params or N>=1."
            raise ValueError(msg)
        
        # Deal with gamma values -----------------------------------------------
        if gamma is not None:
            # Set gamma, check if list or single value
            if hasattr(gamma, '__iter__'):
                gam = gamma
            elif type(gamma) is float:
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

    
    def _compute_rho_sig(self, X, Z, phihat):
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
        a,b = self._pair_idx[:,0], self._pair_idx[:,1]

        numerator = np.einsum('ij,ij->i',X[a],(phihat*X[b]))
        Zphi = phihat*Z
        denominator = np.einsum('ijk,ikj->i',Zphi[a],Zphi[b])

        rho_ab = numerator/denominator
        sig_ab = 1/np.sqrt(denominator)

        if np.isnan(sig_ab).any():
            print(os_ex.NaNPairwiseError.extended_response())
            raise ValueError('NaN values pair-wise uncertainties! Are params valid?')

        return rho_ab, sig_ab


    def _compute_rhok_sigk(self, X, Z, phi, narrowband, select_freq=None):
        """Compute the rho_ab(f_k), sigma_ab(f_k), and normalization_ab(f_k)

        For Internal Use of the OptimalStatistic. Users are not recommended to use this!

        This method calculates the rho_ab(f_k), sigma_ab(f_k), normalization_ab(f_k) 
        for each pulsar pair at each PTA frequency. Details of this implementation
        can be found in Gersbach et al. 2024. Note that the shape of the returned
        matrices are [n_freq x n_pair].

        You can select to use the narrowband-normalization instead of the default
        broadband-normalization by setting the 'narrowband' argument to True.

        You can also select a specific frequency to compute the pair values at by
        setting the 'select_freq' argument to the index of the frequency you wish
        to compute the pair values at. This can be useful if you only care about
        a specific frequency. Setting this to None will compute the pair values at
        all frequencies.

        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phi (numpy.ndarray): A vector of the diagonal phi matrix.
            narrowband (bool): Whether to use the narrowband-normalization instead
                    of the default broadband-normalization.
            select_freq (int, optional): The index of the frequency to compute the
                    pair values at. Defaults to None.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray: rho_ab(f_k), sigma_ab(f_k), 
                                                         normalization_ab(f_k)
        """

        # Compute rho_ab(f_k) for all f_k
        a,b = self._pair_idx[:,0], self._pair_idx[:,1]

        if select_freq is not None:
            phi_til = np.zeros(self.nfreq)  
            phi_til[select_freq] = 1
            phi_til = np.repeat(phi_til,2)
            phi2 = phi/phi[2*select_freq]

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

        rho_abk = np.zeros( (self.nfreq,self.npairs) )
        sig_abk = np.zeros( (self.nfreq,self.npairs) )
        norms_abk = np.zeros( (self.nfreq,self.npairs) )

        for k in range(self.nfreq):
            phi_til = np.zeros(self.nfreq)  
            phi_til[k] = 1
            phi_til = np.repeat(phi_til,2)
            # Use 2*k as there is a sine and cosine power
            phi2 = phi/phi[2*k]

            if narrowband:
                norms_abk[k] = 1/(np.einsum('ijk,ikj->i',phi_til*Z[a],phi_til*Z[b]))
            else:
                norms_abk[k] = 1/(np.einsum('ijk,ikj->i',phi_til*Z[a],phi2*Z[b]))

            rho_abk[k] =  np.sum(X[a] * phi_til * X[b], axis=1) * norms_abk[k]
            sig_abk[k] =  np.sqrt(np.einsum('ijk,ikj->i', phi_til*Z[a], phi_til*Z[b]) * norms_abk[k]**2)

            if np.isnan(sig_abk[k]).any():
                print(os_ex.NaNPairwiseError.extended_response())
                raise os_ex.NaNPairwiseError('NaN values pair-wise uncertainties! Are params valid?')

        return rho_abk, sig_abk, norms_abk


def _get_F_matrices(pta, gwb_name, idx=None):
    """A function to get the F matrices for all pulsars

    Since getting the F matrices can be enterprise version dependent, this
    helper function is used as a quick way to get these matrices and handle 
    that version dependency.

    If idx is None, then this function will return the F matrices for all pulsars.
    
    Args:
        pta (enterprise.signals.signal_base.PTA): A PTA object.
        gwb_name (str): The name of the GWB signal in each pulsar.
        idx (int, optional): The index of the pulsar to get the F matrix for. 
            Defaults to None.
    
    Returns:
        list: A list of F matrices for each pulsar or a single F matrix
            if idx is not None.
    """
    F = []
    try:
        # Some versions of enterprise let you script the pulsar signal
        if idx is None:
            F = [psrsig[gwb_name].get_basis() for psrsig in pta._signalcollections]
        else:
            F = pta._signalcollections[idx][gwb_name].get_basis()

    except:
        # And some don't
        if idx is None:
            for psrsig in pta._signalcollections:
                for sig in psrsig.signals:
                    if sig.signal_id == gwb_name:
                        F.append(sig.get_basis())
                        break
            
            if len(F)!=len(pta._signalcollections):
                raise ValueError('No GWB signal found in PTA._signalcollections[:].signals!')
        else:
            psrsig = pta._signalcollections[idx]
            for sig in psrsig.signals:
                if sig.signal_id == gwb_name:
                    F = sig.get_basis()
                    break
            if F is None:
                raise ValueError('No GWB signal found in PTA._signalcollections[:].signals!')
            
    return F

def _check_deterministic(pta):
    """A function to check if there are deterministic signals in the PTA object.

    This function checks if there are any deterministic signals in the PTA object
    by sampling the parameter space and checking if the delays are non-zero.
    This function uses the pta.get_delay() method to check for these deterministic 
    signals.

    Args:
        pta (enterprise.signals.signal_base.PTA): A PTA object.
    
    Returns:
        list: A list of booleans indicating if there are deterministic signals
    """
    # To check if there is a non-zero delay, we can use the get_delay method
    # and use a random sample of the parameter space. These can be per-pulsar.
    # such as J1713 DM dips
    rand = {p.name:p.sample() for p in pta.params}

    # get_delay() will return a 0 integer if there is no deterministic signals
    # and an array of non-zeros if there is a deterministic signal.
    delays = [np.sum(psr.get_delay(rand))!=0 for psr in pta._signalcollections]

    return delays


def _solveD(N_obj, right, left):
    """A function to solve for the matrix product (left.T) @ D^{-1} @ right

    This function is designed to solve the linear system of equations for the 
    marginalizing timing model formalism of a PTA. This functionality of different
    left and right matrices does not currently exist in 
    enterprise.signals.gp_signals.MarginalizingNmat.solve(), hence this function.

    Args:
        N_obj (enterprise.signals.gp_signals.MarginalizingNmat): The D matrix object
        right (np.ndarray): The matrix on the right side of the equation
        left (np.ndarray): The matrix on the left side of the equation

    Returns:
        np.ndarray: The matrix product (left.T) @ D^{-1} @ right
    """
    Nmat = N_obj.Nmat
    
    lNr = Nmat.solve(right, left_array=left)
    lDr = lNr - np.tensordot(N_obj.MNF(left), N_obj.MNMMNF(right), (0, 0))
    return lDr
