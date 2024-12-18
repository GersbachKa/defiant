
from . import custom_exceptions as os_ex
from . import utils
from . import pair_covariance as pc 
from . import null_distribution as os_nu
from . import orf_functions

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
                 orfs=['hd'], orf_names=None, pcmc_orf=None, max_chunk=300,
                 clip_z=None):
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
        The clip_z parameter is an experimental feature that can be used to clip the
        minimum eigenvalue of the Z matrix products. This can be useful when the data
        is very noisy and leads to non-positive definite matrices. 
        The clip_z parameter represents the minimum allowed eigenvalue of the Z matrix
        when normalized so that the maximum eigenvalue is 1.0. In this way, you cap
        the maximum condition number for all Z products to 1/clip_z. This generally 
        should be kept near machine precision, i.e. we suggest using 1e-16.

        Setting this value to None will disable the clipping, and should always be
        the default option unless you are experiencing issues like NaNPairwiseError.

        Args:
            psrs (list): A list of enterprise.pulsar.BasePulsar objects.
            pta (enterprise.signals.signal_base.PTA): A PTA object.
            gwb_name (str): The name of the GWB in the PTA object. Defaults to 'gw'.
            corepath (str, optional): The location of a pre-saved Core object.
            core (la_forge.core.Core, optional): A la_forge.core.Core object.
            chain_path (str, optional): The location of an PTMCMC chain.
            chain (np.ndarray, optional): The chain.
            param_names (str or list, optional): The names of the chain parameters.
            orfs (list, optional): An orf name or function or list of orf names or functions. 
                Defaults to ['hd'].
            orf_names (str or list, optional): The names of the corresponding orfs. Set to None
                    for default names.
            pcmc_orf (str or function, optional): The assumed ORF for the pair covariance matrix.
                    when using the MCOS. Defaults to None.
            max_chunk (int, optional): The number of allowed simultaneous matrix products to compute. 
                Defaults to 300.
            clip_z (float, optional): (Experimental) The minimum eigenvalue of the Z matrix products.
                Can be useful with very noisy data. Set to None for no clipping. See notes. Defaults to None.

        Raises:
            TypeError: If the PTA object is not of type 'enterprise.signals.signal_base.PTA'.
            TypeError: If the pulsars in the psrs list are not a list or of type 'enterprise.pulsar.BasePulsar'.
        """
        if type(pta) == ent_sig.signal_base.PTA:
            self.pta = pta
        else:
            raise TypeError("pta supplied is not of type 'enterprise.signals.signal_base.PTA'!")
        try: 
            _ = psrs[0] 
        except TypeError: 
            raise TypeError("psrs list supplied is not able to be indexed")
        if isinstance(psrs[0],BasePulsar):
            self.psrs = psrs
        else:
            raise TypeError("pulsars in psrs list is not of type 'enterprise.pulsar.BasePulsar'!")
        
        self.npsr = len(psrs)
        psr_names = self.pta.pulsars

        self.gwb_name = gwb_name
        
        self.lfcore, self.max_like_params = None, None
        self.set_chain_params(core, core_path, chain_path, chain, param_names)

        self.freqs = utils.get_pta_frequencies(pta,gwb_name)
        self.nfreq = len(self.freqs) 
        
        self.pair_names = [(a,b) for i,a in enumerate(psr_names) for b in psr_names[i+1:]]
        self._pair_idx = np.array([(a,b) for a in range(self.npsr) for b in range(a+1,self.npsr)])
        self.npairs = len(self.pair_names)

        self.norfs = 0
        self.orf_functions = []
        self.orf_design_matrix = None
        self._orf_matrix = None
        self._mcos_orf = None
        self.orf_names = None
        self.nside = 0
        self.lmax = 0
        self.set_orf(orfs, orf_names, pcmc_orf)

        self.nmos_iterations = {}

        self._max_chunk = max_chunk
        
        # Check for marginalizing timing model
        self._marginalizing_timing_model = False
        for s in self.pta._signalcollections[0].signals:
            if 'marginalizing linear timing model' in s.signal_name:
                self._marginalizing_timing_model = True
        
        # Pre-cache matrix quantities
        self._cache = {}
        self._compute_cached_matrices()

        self.clip_z = clip_z
        if clip_z is not None:
            warn("Clipping Z matrix products is an experimental feature. Use with caution.")
        


    def set_chain_params(self, core=None, core_path=None, chain_path=None, 
                         chain=None, param_names=None):
        """A method to add MCMC chains to an OSpluplus object. 

        This method takes a number of different forms of MCMC chains and creates
        a la_forge.core.Core object for use with noise marginalization or maximum 
        likelihood optimal statistic. To use this function, you must include one
        or more of the following options from most prefered to least:
            1. core 
            2. corepath   
            3. chain_path  
            4. chain & param_names   
            5. chain

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
            self.max_like_params = utils.get_max_like_params(self.lfcore)


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
            
        pcmc_orf is an experimental feature that aims to curb the problematic 
        nature of pair covariance matrix with the MCOS. 

        If this argument is set to None (default): 
            - The typical behavior will be used, this being that the
            power per process will be the normalized non-pair covariant MCOS 
            multiplied by the CURN amplitude. 
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

        for i in range( len(orfs) ):
            orf = orfs[i]
            name = orfs[i]

            if type(orf) == str:
                # ORF must be pre-defined function
                cur_orf = orf_functions.get_orf_function(orf)
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
                orf = orf_functions.get_orf_function(pcmc_orf)
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
        orf = orf_functions.get_orf_function(pc_orf)
        temp = np.zeros( (self.npsr,self.npsr) )
        for a in range(self.npsr):
            for b in range(a+1,self.npsr):
                v = orf(self.psrs[a],self.psrs[b])
                temp[a,b] = v
                temp[b,a] = v
        self._mcos_orf = temp

        if basis.lower() == 'pixel':
            basis = orf_functions.anisotropic_pixel_basis(self.psrs, nside, self._pair_idx)
            self.nside = nside
            self.lmax = None

            self.orf_names = [f'pixel_{i}' for i in range(basis.shape[1])]
            self.orf_design_matrix = basis
            self._orf_matrix = None
            self.norfs = basis.shape[1]

        elif basis.lower() == 'spherical':
            basis = orf_functions.anisotropic_spherical_harmonic_basis(self.psrs, lmax, 
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
                   return_pair_vals=True, fisher_diag_only=False, use_tqdm=True):
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

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component OS processes
        simultaneously. Keep this on True unless you know what you are doing.

        Args:
            params (dict, optional): A dictionary of key:value parameters. 
                Defaults to maximum likelihood. Only used if N=1.
            N (int): The number of NMOS iterations to run. If 1, uses params. Defaults to 1.
            gamma (float, optional): The spectral index to use for analysis. If set to None,
                this function first checks if gamma is in params, otherwise it will
                assume the PTA model is a fixed gamma and take it from there. Defaults to None.
            pair_covariance (bool): Whether to use pair covariance. Defaults to False.
            return_pair_vals (bool): Whether to return the xi, rho, sig, C values. Defaults to True.
            fisher_diag_only (bool): Whether to zero the off-diagonal elements of the
                fisher matrix.
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.
            os_ex.NMOSInteruptError: If the noise marginalization iterations are interupted.

        Returns:
            Return values are very different depending on which options you choose.
            
            Values marked with a * are floats while every other return is an np.array.
            If N=1 and return_pair_vals=False:
                - returns A2*, A2S*
            If N=1 and return_pair_vals=True:
                - returns xi, rho, sig, C, A2*, A2S*
            If N>1 and return_pair_vals=False:
                - returns A2, A2S, param_index
            If N>1 and return_pair_vals=True:
                - returns xi, rho, sig, C, A2, A2S, param_index
            
            A2 (np.ndarray/float) - The OS amplitude estimators at 1/yr
            A2s (np.ndarray/float) - The 1-sigma uncertainties of A2
            xi (np.ndarray) - The pair separations of the pulsars
            rho (np.ndarray) - The pair correlated powers
            sig (np.ndarray) - The pair uncertainties in rho
            C (np.ndarray) - The pair covariance matrix (either a vector or matrix)
            param_index (np.ndarray) - The index of the parameter vectors used in NM each iteration
        """
        # TODO: return gamma values used in some form as well!

        if N==1:
            if params is None and self.lfcore is None:
                msg = "No parameters given and no chain files to default to!"
                raise ValueError(msg)
            
            elif params is None and self.lfcore is not None:
                msg = 'No parameters set without noise marginalization, defaulting '+\
                      'to maximum likelihood OS.'
                warn(msg)

                params = self.max_like_params
                
            pars = utils.freespec_param_fix(params,self.gwb_name)

            # Calculating the phihat
            if gamma is not None:
                # Set gamma
                phihat = utils.powerlaw(self.freqs, 0, gamma, 2)
            elif self.gwb_name+'_gamma' in pars:
                # Varied gamma
                g = pars[self.gwb_name+'_gamma']
                phihat = utils.powerlaw(self.freqs, 0, g, 2)
            else:
                # Fixed gamma, not supplied
                g = utils.get_fixed_gwb_gamma(self.pta, self.gwb_name)
                phihat = utils.powerlaw(self.freqs, 0, g, 2)

            rho, sig, C, A2, A2s = self._compute_os_iteration(pars, phihat, pair_covariance, 
                                                              fisher_diag_only, use_tqdm)
            
            if return_pair_vals:
                xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
                return xi, rho, sig, C, A2, A2s
            else:
                return A2, A2s
        
        # Noise marginalized            
        if self.lfcore is None:
            msg = 'Cannot Noise marginalize without a La forge core!'
            raise ValueError(msg)

        
        self.nmos_iterations = {'A2':[],'A2s':[],'param_index':[]}
        if return_pair_vals:
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rho'] = []
            self.nmos_iterations['sig'] = []
            self.nmos_iterations['C'] = []
        try:
            iterable = tqdm(range(N),desc='NM Iterations') if use_tqdm  else range(N)
        
            for iter in iterable:
                rand_i = np.random.random_integers(self.lfcore.burn, self.lfcore.chain.shape[0]-1)
                params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.chain[rand_i])}
                
                pars = utils.freespec_param_fix(params,self.gwb_name)

                # Calculating the phihat
                if gamma is not None:
                    # Set gamma
                    phihat = utils.powerlaw(self.freqs, 0, gamma, 2)
                elif self.gwb_name+'_gamma' in pars:
                    # Varied gamma
                    g = pars[self.gwb_name+'_gamma']
                    phihat = utils.powerlaw(self.freqs, 0, g, 2)
                else:
                    # Fixed gamma, not supplied
                    g = utils.get_fixed_gwb_gamma(self.pta, self.gwb_name)
                    phihat = utils.powerlaw(self.freqs, 0, g, 2)
                
                rho,sig,C,A2,A2s = self._compute_os_iteration(pars, phihat, pair_covariance, 
                                                              fisher_diag_only, use_tqdm)

                self.nmos_iterations['A2'].append(A2)
                self.nmos_iterations['A2s'].append(A2s)
                self.nmos_iterations['param_index'].append(rand_i)
                
                if return_pair_vals:
                    self.nmos_iterations['rho'].append(rho)
                    self.nmos_iterations['sig'].append(sig)
                    self.nmos_iterations['C'].append(C)

        except Exception as e:
            msg = 'Stopping NMOS iterations. Calculated values are can be found '+\
                  'in OptimalStatistic.nmos_iterations.'
            raise os_ex.NMOSInteruptError(msg) from e

        
        A2 = np.array( self.nmos_iterations['A2'] )
        A2s = np.array( self.nmos_iterations['A2s'] )
        param_index = np.array( self.nmos_iterations['param_index'] )

        if return_pair_vals:
            xi = self.nmos_iterations['xi']
            rho = np.array( self.nmos_iterations['rho'] )
            sig = np.array( self.nmos_iterations['sig'] )
            C = np.array( self.nmos_iterations['C'] )
            return xi, rho, sig, C, A2, A2s, param_index

        return A2, A2s, param_index


    def compute_PFOS(self, params=None, N=1, pair_covariance=False, narrowband=False, 
                     return_pair_vals=True, fisher_diag_only=False, use_tqdm=True):
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
        If you are using a varied gamma CURN model, you can either:
            - Set a particular gamma value for all NM PF OS iterations by setting gamma
            - Or set gamma=None and the function will default to each iterations' gamma value

        There is also an option to use only the diagonal elements of the Fisher matrix,
        which can be useful if you are trying to measure many single component OS processes
        simultaneously. Keep this on True unless you know what you are doing.

        Args:
            params (dict, optional): A dictionary of key:value parameters. 
                Defaults to maximum likelihood. Only used if N=1.
            N (int): The number of NM PF OS iterations to run. If 1, uses params. Defaults to 1.
            pair_covariance (bool): Whether to use pair covariance. Defaults to False.
            narrowband (bool): Whether to use the narrowband-normalized PFOS instead of
                the default broadband-normalized PFOS. Defaults to False.
            return_pair_vals (bool): Whether to return the xi, rhok, sigk, Ck values. Defaults to True.
            fisher_diag_only (bool): Whether to zero the off-diagonal elements of the
                fisher matrix.
            use_tqdm (bool): Whether to use a progress bar. Defaults to True.

        Raises:
            ValueError: If params is None and to la_forge core is set.
            ValueError: If Noise Marginalization is attempted without a La forge core.
            os_ex.NMOSInteruptError: If the noise marginalization iterations are interupted.

        Returns:
            Return values are very different depending on which options you choose.
            All values are np.ndarrays.
            If N=1 and return_pair_vals=False:
                - returns Sk, Sks
            If N=1 and return_pair_vals=True:
                - returns xi, rhok, sigk, Ck, Sk, Sks
            If N>1 and return_pair_vals=False:
                - returns Sk, Sks, param_index
            If N>1 and return_pair_vals=True:
                - returns xi, rhok, sigk, Ck, Sk, Sks, param_index
            
            Sk (np.ndarray) - The PFOS S(f_k) [yr^2] for each frequency number k
            Sks (np.ndarray) - The 1-sigma uncertainties for Sk
            xi (np.ndarray) - The pair separations of the pulsars
            rhok (np.ndarray) - The pair correlated PSD for each frequency number k
            sigk (np.ndarray) - The pair uncertainties in rhok
            Ck (np.ndarray) - The pair covariance matrix for each frequency number k
            param_index (np.ndarray) - The index of the parameter vectors used in each NM iteration
        """

        if N==1:
            if params is None and self.lfcore is None:
                msg = "No parameters given and no chain files to default to!"
                raise ValueError(msg)
            
            elif params is None and self.lfcore is not None:
                msg = 'No parameters set without noise marginalization, defaulting '+\
                      'to maximum likelihood OS.'
                warn(msg)

                params = self.max_like_params

            pars = utils.freespec_param_fix(params,self.gwb_name)
            rhok,sigk,Ck,Sk,Sks = self._compute_pfos_iteration(pars, narrowband, 
                                            pair_covariance, fisher_diag_only, use_tqdm)
            
            if return_pair_vals:
                xi,_ = utils.compute_pulsar_pair_separations(self.psrs,self._pair_idx)
                return xi,rhok,sigk,Ck,Sk,Sks
            else:
                return Sk,Sks

        # Noise marginalized            
        if self.lfcore is None:
            msg = 'Cannot Noise marginalize with a Null La forge core! '
            raise ValueError(msg)
        
                
        self.nmos_iterations = {'Sk':[],'Sks':[],'param_index':[]}
        if return_pair_vals:
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rhok'] = []
            self.nmos_iterations['sigk'] = []
            self.nmos_iterations['Ck'] = []

        try:
            iterable = tqdm(range(N),desc='NM Iterations') if use_tqdm else range(N)

            for iter in iterable:
                rand_i = np.random.random_integers(self.lfcore.burn, self.lfcore.chain.shape[0]-1)
                params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.chain[rand_i])}
                pars = utils.freespec_param_fix(params, self.gwb_name)
                
                rhok,sigk,Ck,Sk,Sks = self._compute_pfos_iteration(pars, narrowband, pair_covariance, 
                                                                   fisher_diag_only, use_tqdm)
                        
                self.nmos_iterations['Sk'].append(Sk)
                self.nmos_iterations['Sks'].append(Sks)
                self.nmos_iterations['param_index'].append(rand_i)
                if return_pair_vals:
                    self.nmos_iterations['rhok'].append(rhok)
                    self.nmos_iterations['sigk'].append(sigk)
                    self.nmos_iterations['Ck'].append(Ck)

        except Exception as e:
            msg = 'Stopping NMOS iterations. Calculated values are can be found in '+\
                  'Optimal_statistic.nmos_iterations.'
            raise os_ex.NMOSInteruptError(msg) from e

        
        Sk = np.array( self.nmos_iterations['Sk'] )
        Sks = np.array( self.nmos_iterations['Sks'] )
        param_index = np.array( self.nmos_iterations['param_index'] )

        if return_pair_vals:
            xi = np.array( self.nmos_iterations['xi'] )
            rhok = np.array( self.nmos_iterations['rhok'] )
            sigk = np.array( self.nmos_iterations['sigk'] )
            Ck = np.array( self.nmos_iterations['Ck'] )
            return xi,rhok, sigk, Ck, Sk, Sks, param_index
    
        return Sk, Sks, param_index
    

    def _compute_cached_matrices(self):
        """A function to compute the constant valued matrices used in the OS.

        This function will calculate the following matrix product, which are constant
        for any parameter values given to the PTA.
        The stored matrix products are:
            - FNr, FNF, FNT, TNT, TNr

        Note: If the PTA is using the marginalizing timing model T will be replaced
        with F_irn, where F is the fourier design matrix of GWB only, and F_irn is
        the fourier design matrix of the full red noise.
        """
        all_FNr = []
        all_FNF = []
        all_FNT = [] 
        all_TNT = []
        all_TNr = []

        # Getting the F matrices are enterprise version dependent. Move to a function
        all_F = self._get_F_matrices()

        for idx,psr_signal in enumerate(self.pta._signalcollections):
            r = psr_signal._residuals
            N = psr_signal.get_ndiag()
            T = psr_signal.get_basis()
            F = all_F[idx]
            
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

            FNr = self._cache['FNr'][a]
            FNF = self._cache['FNF'][a]
            FNT = self._cache['FNT'][a]
            TNT = self._cache['TNT'][a]
            TNr = self._cache['TNr'][a]

            sigma = phiinv + TNT
            try:
                cf = sl.cho_factor(sigma)
                X[a] = FNr - FNT @ sl.cho_solve(cf, TNr)
                Z[a] = FNF - FNT @ sl.cho_solve(cf, FNT.T)
            except np.linalg.LinAlgError:
                X[a] = FNr - FNT @ np.linalg.solve(sigma, TNr)
                Z[a] = FNF - FNT @ np.linalg.solve(sigma, FNT.T)

        if self.clip_z is not None:
            for i in range(len(Z)):
                Z[i] = utils.clip_covariance(Z[i], self.clip_z)

        return X, Z
    

    def _compute_os_iteration(self, params, phihat, pair_covariance, fisher_diag_only, 
                              use_tqdm):
        """Compute a single iteration of the OS. Users should use compute_OS() instead.

        An internal function to run a single iteration of the optimal statistic 
        using the supplied parameter dictionary. This function will give 1-sigma 
        uncertainties for A2s if there is a single ORF, otherwise it will return 
        the covaraince matrix between the processes.

        Args:
            params (dict): A dictionary of parameter values for the PTA.
            phihat (numpy.ndarray): The unit-model spectrum to evaluate the OS with.
            pair_covariance (bool): Whether to use pair covariance in the solving.
            fisher_diag_only (bool): Whether to zero the off-diagonal elements of the
                fisher matrix.
            use_tqdm (bool): Whether to use a progress bar.

        Returns:
            rho_ab (numpy.ndarray): The pairwise correlated powers.
            sig_ab (numpy.ndarray): The pairwise uncertainties in rho_ab.
            C (numpy.ndarray): The pair covariance matrix used.
            A2 (numpy.ndarray): The \hat{A}^2 of the GWB.
            A2s (numpy.ndarray): The uncertainty or covariance matrix in Sk for each frequency.
        """
        X,Z = self._compute_XZ(params)
        rho_ab, sig_ab = self._compute_rho_sig(X,Z,phihat)

        if pair_covariance:
            solve_method='woodbury'

            if self.lfcore is not None and self.gwb_name+'_log10_rho_0' in self.lfcore.params:
                param_cov = utils.freespec_covariance(self.lfcore,self.gwb_name)
            else:
                param_cov = np.diag(np.ones_like(self.freqs))

            a2_est = utils.fit_a2_from_params(params, gwb_name=self.gwb_name, model_phi=phihat[::2], cov=param_cov)

            if self.norfs>1:
                # MCOS
                if self._mcos_orf is not None:
                    # The user has set an assumed MCOS ORF
                    C = pc._compute_pair_covariance(Z, phihat, phihat, self._mcos_orf,
                            np.square(sig_ab), a2_est, use_tqdm, self._max_chunk)
                else:
                    # Default behavior
                    C = pc._compute_mcos_pair_covariance(Z, phihat, phihat, 
                            self._orf_matrix, self.orf_design_matrix, rho_ab, sig_ab, 
                            np.square(sig_ab), a2_est, use_tqdm, self._max_chunk)
            else:
                # Single component
                # Splits the covariance matrix into diagonal and off-diagonal elements
                C = pc._compute_pair_covariance(Z, phihat, phihat, 
                        self._orf_matrix[0], np.square(sig_ab), a2_est, use_tqdm, self._max_chunk)
             
        else:
            solve_method='diagonal'
            C = np.square(sig_ab)
        
        A2, A2s = utils.linear_solve(self.orf_design_matrix, C, rho_ab[:,None], 
                                                solve_method, fisher_diag_only)
        
        A2 = np.squeeze(A2) if self.norfs>1 else A2.item()
        A2s = np.squeeze(A2s) if self.norfs>1 else np.sqrt(A2s.item())

        if pair_covariance:
            C = C[0]+C[1]

        return rho_ab, sig_ab, C, A2, A2s


    def _compute_pfos_iteration(self, params, narrowband, pair_covariance, fisher_diag_only, 
                                use_tqdm):
        """Compute a single iteration of the PFOS. Users should use compute_PFOS() instead.

        An internal function to run a single iteration of the per frequency 
        optimal statistic using the supplied parameter dictionary. This function
        defaults to using the broadband-normalized PFOS. This function will give
        1-sigma uncertainties for Sks if there is a single ORF, otherwise it will
        return the covaraince matrix between the processes. Details of the PFOS 
        implementation can be found in Gersbach et al. 2024.

        Args:
            params (dict): A dictionary of parameter values for the PTA.
            narrowband (bool): Whether to use the narrowband-normalized PFOS instead.
            pair_covariance (bool): Whether to use pair covariance in the solving.
            fisher_diag_only (bool): Whether to zero the off-diagonal elements of the
                fisher matrix.
            use_tqdm (bool): Whether to use a progress bar.

        Returns:
            rho_abk (numpy.ndarray): The pairwise correlated powers for each frequency.
            sig_abk (numpy.ndarray): The pairwise uncertainties in rho_abk for each frequency.
            Ck (numpy.ndarray): The pair covariance matrix used for each frequency.
            Sk (numpy.ndarray): The PSD(f_k)/T_{span} for each frequency.
            Sks (numpy.ndarray): The uncertainty or covariance matrix in Sk for each frequency.
        """
        X,Z = self._compute_XZ(params)
        gw_signal = [s for s in self.pta._signalcollections[0].signals if s.signal_id==self.gwb_name][0]
        phi = gw_signal.get_phi(params)
        rho_abk, sig_abk, norm_abk = self._compute_rhok_sigk(X,Z,phi,narrowband)

        Sk = np.zeros( (self.nfreq, self.norfs) )
        Sks = np.zeros( (self.nfreq, self.norfs,self.norfs) )
        if pair_covariance:
            Ck = np.zeros( (self.nfreq,self.npairs,self.npairs) )
        else:
            Ck = np.zeros( (self.nfreq,self.npairs) )

        iterable = tqdm(range(self.nfreq),desc='Frequencies',leave=False) \
            if use_tqdm else range(self.nfreq)

        for k in iterable:
            sk = phi[2*k]

            phi1 = np.zeros( shape=(len(self.freqs)) )  
            phi1[k] = 1
            phi1 = np.repeat(phi1,2)

            phi2 = phi / sk

            if pair_covariance:
                solve_method = 'woodbury'
                if self.norfs>1:
                    # MCOS
                    if self._mcos_orf is not None:
                        # The user has set an assumed MCOS ORF
                        C = pc._compute_pair_covariance(Z, phi1, phi2, self._mcos_orf,
                                norm_abk[k], sk, use_tqdm, self._max_chunk)
                    else:
                        # Default behavior
                        C = pc._compute_mcos_pair_covariance(Z, phi1, phi2, self._orf_matrix, 
                                self.orf_design_matrix, rho_abk[k], sig_abk[k], norm_abk[k], sk,
                                use_tqdm, self._max_chunk)
                else:
                    # Single component
                    C = pc._compute_pair_covariance(Z, phi1, phi2, 
                            self._orf_matrix[0], norm_abk[k], sk, use_tqdm, self._max_chunk)
            else:
                solve_method='diagonal'
                C = sig_abk[k]**2
        
            s, ssig = utils.linear_solve(self.orf_design_matrix, C, rho_abk[k,:,None], 
                                         solve_method, fisher_diag_only)
            
            if pair_covariance:
                Ck[k] = C[0]+C[1]
            else:
                Ck[k] = C
            Sk[k] = np.squeeze(s) if self.norfs>1 else s.item()
            Sks[k] = np.squeeze(ssig) if self.norfs>1 else np.sqrt(ssig.item()) 

        Sk = np.squeeze(Sk) 
        Sks = np.squeeze(Sks)
        Ck = np.squeeze(Ck)

        return rho_abk, sig_abk, Ck, Sk, Sks

    
    def _compute_rho_sig(self, X, Z, phihat):
        """Compute the rho_ab, sigma_ab correlation and uncertainties

        For Internal Use of the OptimalStatistic. Users are not recommended to use this!

        This method calculates the rho_ab and sigma_ab for each pulsar pair using
        the OS' X, Z and phihat matrix products. Check Appendix A from 
        Pol et al. 2023, or Gersbach et al. 2024 for matrix product definitions.

        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phihat (numpy.ndarray): A vector of the diagonal \hat{\phi} matrix.

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


    def _compute_rhok_sigk(self, X, Z, phi, narrowband):
        """Compute the rho_ab(f_k), sigma_ab(f_k), and normalization_ab(f_k)

        For Internal Use of the OptimalStatistic. Users are not recommended to use this!

        This method calculates the rho_ab(f_k), sigma_ab(f_k), normalization_ab(f_k) 
        for each pulsar pair at each PTA frequency. Details of this implementation
        can be found in Gersbach et al. 2024. Note that the shape of the returned
        matrices are [n_freq x n_pair].


        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phi (numpy.ndarray): A vector of the diagonal \phi matrix.
            narrowband (bool): Whether to use the narrowband-normalization instead
                    of the default broadband-normalization.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray: rho_ab(f_k), sigma_ab(f_k), 
                                                         normalization_ab(f_k)
        """

        # Compute rho_ab(f_k) for all f_k
        a,b = self._pair_idx[:,0], self._pair_idx[:,1]

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


    def _get_F_matrices(self):
        """A function to get the F matrices for all pulsars

        Since getting the F matrices can be enterprise version dependent, this
        helper function is used as a quick way to get these matrices and handle 
        that version dependency.

        Returns:
            list: A list of F matrices for each pulsar
        """
        F = []
        try:
            # Some versions of enterprise let you script the pulsar signal
            F = [psrsig[self.gwb_name].get_basis() for psrsig in self.pta._signalcollections]
        except:
            # And some don't
            for psrsig in self.pta._signalcollections:
                for sig in psrsig.signals:
                    if sig.signal_id == self.gwb_name:
                        F.append(sig.get_basis())
                        break
            
            if len(F)!=len(self.pta._signalcollections):
                raise ValueError('No GWB signal found in PTA._signalcollections[0].signals!')
        return F


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
