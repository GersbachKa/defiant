
from . import ospp_exceptions as os_ex
from . import utils
from . import pair_covariance as pc 
from . import ospp_nulldist as os_nu

import numpy as np
import scipy.linalg as sl

import enterprise.signals as ent_sig
from enterprise.pulsar import BasePulsar

from enterprise.signals.utils import powerlaw
from enterprise_extensions import model_orfs
from la_forge.core import Core

from tqdm import tqdm
from warnings import warn


class OSplusplus:


    def __init__(self, psrs, pta, gwb_name='gw', corepath=None, core=None,  
                 chain_path=None, chain=None, param_names=None, 
                 orfs=['hd'], orf_names=None, max_chunk=300):
        
        # Order of psrs needs to be the same as the one given to pta
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
        self.set_chain_params(core, corepath, chain_path, chain, param_names)

        self.freqs = utils.get_pta_frequencies(pta,gwb_name)
        self.nfreq = len(self.freqs) 
        
        self.pair_names = [(a,b) for i,a in enumerate(psr_names) for b in psr_names[i+1:]]
        self._pair_idx = np.array([(a,b) for a in range(self.npsr) for b in range(a+1,self.npsr)])
        self.npairs = len(self.pair_names)

        self.norfs = 1
        self.orf_design_matrix, self._orf_matrix, self.orf_names = None, None, None
        self.set_orf(orfs, orf_names)

        self.nmos_iterations = {}

        self._max_chunk = max_chunk

        self._cache = {}


    def set_chain_params(self, core=None, corepath=None, chain_path=None, 
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
        elif corepath is not None:
            self.lfcore = Core(corepath=corepath)
        elif chain_path is not None:
            self.lfcore = Core(chaindir=chain_path)
        elif chain is not None and param_names is not None:
            self.lfcore = Core(chain=chain,params=param_names)
        elif chain is not None:
            self.lfcore = Core(chain=chain,params=self.pta.param_names)
        else:
            msg = 'No MCMC samples were given! Set these later or supply ' +\
                  'them when computing the OS.'
            warn(msg)

        if self.lfcore is not None:
            self.max_like_params = utils.get_max_like_params(self.lfcore)


    def set_orf(self, orfs=['hd'], orf_names=None):
        """Sets the overlap reduction function[s] (ORF) for the cross correlations.

        Sets the overlap reduction function[s] of the cross correlation and the
        corresponding ORF design matrix. This function supports multiple ORFs by
        simply supplying a list of ORFs. orf_names can be left as None to use default
        names. orfs can also be a user-defined function which accepts 2 
        enterprise.pulsar.BasePulsar. Otherwise, use one of the following pre-defined
        ORF names:
            'hd' - Hellings and Downs
            'dipole' - Dipole
            'monopole' - Monopole
            'gw_dipole' - Gravitational wave dipole
            'gw_monopole' - Gravitational wave monopole
            'st' - Scalar transverse

        Args:
            orfs (list, optional): An ORF string, list of strings, or function. 
                    Note that a custom function must accept two enterprise.pulsar.BasePulsar
                    objects as inputs and outputs a float for their ORF.
            orf_names (list, optional): The names of the corresponding orfs. Set to None
                    for default names.

        Raises:
            ValueError: If the length of the orfs and orf_names does not match.
            NameError: If a pre-defined ORF is not found.
            TypeError: If the user-supplied ORF does not have correct format. 
        """
        if not hasattr(orfs, '__getitem__'):
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
        # Used for pair covariance
        self._orf_matrix = np.ones( (self.norfs,self.npsr,self.npsr) ) 

        for i in range( len(orfs) ):
            orf = orfs[i]
            name = orf[i]

            if type(orf) == str:
                # ORF must be one of the built-in functions
                if orf.lower() == 'hd':
                    cur_orf = model_orfs.hd_orf
                    name = 'HD' if name is None else name
                elif orf.lower() == 'dipole':
                    cur_orf = model_orfs.dipole_orf
                    name = 'Dipole' if name is None else name
                elif orf.lower() == 'monopole':
                    cur_orf = model_orfs.monopole_orf
                    name = 'Monopole' if name is None else name
                elif orf.lower() == 'gw_dipole':
                    cur_orf = model_orfs.gw_dipole_orf
                    name = 'GW_Dipole' if name is None else name
                elif orf.lower() == 'gw_monopole':
                    cur_orf = model_orfs.gw_monopole_orf
                    name = 'GW_Monopole' if name is None else name
                elif orf.lower() == 'st':
                    cur_orf = model_orfs.st_orf
                    name = 'Scalar transpose' if name is None else name
                else:
                    msg = f"Unknown ORF name: '{orf}'. Check the documentation " +\
                           "for pre-programmed ORFs or supply your own."
                    raise NameError(msg)
                
                self.orf_names.append(name)
                for j,(a,b) in enumerate(self._pair_idx):
                    v = cur_orf(self.psrs[a].pos , self.psrs[b].pos)
                    self.orf_design_matrix[i,j] = v
                    self._orf_matrix[i,a,b] = v
                    self._orf_matrix[i,b,a] = v

            else:
                # ORF is user supplied function
                name = orf.__name__ if name is None else name
                self.orf_names.append(name)
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
        

    def compute_OS(self, params=None, N=1, gamma=None, pair_covariance=True, 
                   return_pair_vals=True, use_tqdm=True):
        
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
            if gamma is None and self.gwb_name+'_gamma' in params:
                gamma = params[self.gwb_name+'_gamma']
            else:
                gamma = utils.get_fixed_gwb_gamma(self.pta, self.gwb_name)
            phihat = powerlaw(np.repeat(self.freqs,2), 0, gamma)
            
            xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)

            rho, sig, C, A2, A2s = self._compute_os_iteration(pars, phihat,
                                                    pair_covariance, use_tqdm)
            
            if return_pair_vals:
                xi,_ = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
                return xi,rho,sig,C,A2,A2s
            else:
                return A2,A2s
        
        # Noise marginalized            
        if self.lfcore is None:
            msg = 'Cannot Noise marginalize without a La forge core!'
            raise ValueError(msg)

        
        self.nmos_iterations = {'A2':[],'A2s':[],'param_index':[]}
        if return_pair_vals:
            xi = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rho'] = []
            self.nmos_iterations['sig'] = []
            self.nmos_iterations['C'] = []
        try:
            # I would parallelize this, but unfortunately I can't pickle an OS++ object
            # due to the pta requirement. If I could pickle a PTA, it would be trivial!
            for iter in tqdm(range(N)) if use_tqdm else range(N):
                rand_i = np.random.random_integers(self.lfcore.burn, self.lfcore.chain.shape[0]-1)
                params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.chain[rand_i])}
                
                pars = utils.freespec_param_fix(params,self.gwb_name)
                if gamma is None and self.gwb_name+'_gamma' in params:
                    gamma = params[self.gwb_name+'_gamma']
                else:
                    gamma = utils.get_fixed_gwb_gamma(self.pta, self.gwb_name)
                phihat = powerlaw(np.repeat(self.freqs,2), 0, gamma)

                rho,sig,C,A2,A2s = self._compute_os_iteration(pars, phihat, 
                                                        pair_covariance, False)

                self.nmos_iterations['A2'].append(A2)
                self.nmos_iterations['A2s'].append(A2s)
                self.nmos_iterations['param_index'].append(rand_i)
                
                if return_pair_vals:
                    self.nmos_iterations['rho'].append(rho)
                    self.nmos_iterations['sig'].append(sig)
                    self.nmos_iterations['C'].append(C)

        except Exception as e:
            msg = 'Stopping NMOS iterations. Calculated values are can be found in OSplusplus.nmos_iterations.'
            raise os_ex.NMOSInteruptError(msg) from e

        
        A2 = np.array( self.nmos_iterations['A2'] )
        A2s = np.array( self.nmos_iterations['A2s'] )
        param_index = np.array( self.nmos_iterations['param_index'] )

        if return_pair_vals:
            xi = np.array( self.nmos_iterations['xi'] )
            rho = np.array( self.nmos_iterations['rho'] )
            sig = np.array( self.nmos_iterations['sig'] )
            C = np.array( self.nmos_iterations['C'] )
            return np.array( xi,rho,sig,C,A2,A2s,param_index )

        return A2, A2s, param_index


    def compute_PFOS(self, params=None, N=1, pair_covariance=True, narrowband=False,
                     return_pair_vals=True, use_tqdm=True):
        
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
                                                        pair_covariance, use_tqdm)
            
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
            xi = utils.compute_pulsar_pair_separations(self.psrs, self._pair_idx)
            self.nmos_iterations['xi'] = xi
            self.nmos_iterations['rhok'] = []
            self.nmos_iterations['sigk'] = []
            self.nmos_iterations['Ck'] = []

        try:
            # I would parallelize this, but unfortunately I can't pickle an OS++ object
            # due to the pta requirement. If I could pickle a PTA, it would be trivial!
            for iter in tqdm(range(N)) if use_tqdm else range(N):
                rand_i = np.random.random_integers(self.lfcore.burn, self.lfcore.chain.shape[0]-1)
                params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.chain[rand_i])}
                pars = utils.freespec_param_fix(params, self.gwb_name)
                
                rhok,sigk,Ck,Sk,Sks = self._compute_pfos_iteration(pars, narrowband,
                                                            pair_covariance, False)
                        
                self.nmos_iterations['Sk'].append(Sk)
                self.nmos_iterations['Sks'].append(Sks)
                self.nmos_iterations['param_index'].append(rand_i)
                if return_pair_vals:
                    self.nmos_iterations['rhok'].append(rhok)
                    self.nmos_iterations['sigk'].append(sigk)
                    self.nmos_iterations['Ck'].append(Ck)

        except Exception as e:
            msg = 'Stopping NMOS iterations. Calculated values are can be found in OSplusplus.nmos_iterations.'
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
    

    def _compute_XZ(self, params):
        """A function to quickly calculate the OS' matrix quantities

        This function calculates the X and Z matrix quantities from the appendix A
        of Pol, Taylor, Romano, 2022: (https://arxiv.org/abs/2206.09936). X and Z
        can be represented as X = F^T @ P^{-1} @ r and Z = F^T @ P^{-1} @ F.

        Args:
            params (dict): A dictionary containing the parameter name:value pairs for the PTA
            use_cache (bool): Whether to use cached values for constant matrix products.

        Returns:
            (np.array, np.array): A tuple of X and Z. X is an array of vectors for each pulsar 
                (N_pulsar x 2N_frequency). Z is an array of matrices for each pulsar 
                (N_pulsar x 2N_frequency x 2N_frequency)
        """
        # TODO: Deal with caching values. Which matrix products can be cached between
        # parameter changes?

        X = np.zeros( shape = ( self.npsr, 2*self.nfreq ) ) # An array of vectors
        Z = np.zeros( shape = ( self.npsr, 2*self.nfreq, 2*self.nfreq ) ) # An array of matrices

        for a,psr_signal in enumerate(self.pta):
            # Need residuals r, GWB Fourier design F, and pulsar design matrix T = [M F]
            r = psr_signal._residuals
            F = psr_signal[self.gwb_name].get_basis(params)
            T = psr_signal.get_basis(params)

            # Used in creating P^{-1}
            # Need N, use .solve() for inversions
            N = psr_signal.get_ndiag(params)

            # sigma = B^{-1} + T^T @ N^{-1} @ T
            sigma = sl.cho_factor( np.diag(psr_signal.get_phiinv(params)) + psr_signal.get_TNT(params) )

            FNr = N.solve(r,F) # F^T @ N^{-1} @ r
            TNr = N.solve(r,T) # T^T @ N^{-1} @ r
            FNT = N.solve(T,F) # F^T @ N^{-1} @ T
            FNF = N.solve(F,F) # F^T @ N^{-1} @ F
        
            # X = F^T @ P^{-1} @ r =
            # F^T @ N^{-1} @ r - F^T @ N^{-1} @ T @ sigma^{-1} @ T^T @ N^{-1} @ r
            X[a] = FNr - FNT @ sl.cho_solve(sigma, TNr)

            # Z = F^T @ P^{-1} @ F =
            # F^T @ N^{-1} @ F - F^T @ N^{-1} @ T @ sigma^{-1} @ T^T @ N^{-1} @ F
            Z[a] = FNF - FNT @ sl.cho_solve(sigma, FNT.T)

        return X, Z
    

    def _compute_os_iteration(self, params, phihat, pair_covariance, use_tqdm):
        """Compute a single iteration of the OS. Users should use compute_OS() instead.

        An internal function to run a single iteration of the optimal statistic 
        using the supplied parameter dictionary. This function will give 1-sigma 
        uncertainties for A2s if there is a single ORF, otherwise it will return 
        the covaraince matrix between the processes.

        Args:
            params (dict): A dictionary of parameter values for the PTA.
            phihat (numpy.ndarray): The unit-model spectrum to evaluate the OS with.
            pair_covariance (bool): Whether to use pair covariance in the solving.
            use_tqdm (bool): Whether to use a TQDM progress bar.

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
            solve_method='pinv'
            if self.lfcore is not None and self.gwb_name+'_log10_rho_0' in self.lfcore.params:
                param_cov = utils.freespec_covariance(self.lfcore,self.gwb_name)
            else:
                param_cov = np.diag(np.ones_like(self.freqs))

            a2_est = utils.fit_a2_from_params(params, model_phi=phihat[::2], cov=param_cov)

            if self.norfs>1:
                # MCOS
                C = pc._compute_mcos_pair_covariance(Z, phihat, phihat, 
                        self._orf_matrix, self.orf_design_matrix, rho_ab, sig_ab, 
                        np.square(sig_ab), a2_est, use_tqdm, self._max_chunk)
            else:
                # Single component
                C = pc._compute_pair_covariance(Z, phihat, phihat, 
                        self._orf_matrix[0], np.square(sig_ab), a2_est, use_tqdm, self._max_chunk)
        else:
            solve_method='diagonal'
            C = np.square(sig_ab)
        
        A2, A2s = utils.linear_solve(self.orf_design_matrix, C, rho_ab[:,None], 
                                                solve_method)
        
        A2 = np.squeeze(A2) if self.norfs>1 else A2.item()
        A2s = np.squeeze(A2s) if self.norfs>1 else np.sqrt(A2s.item())

        return rho_ab, sig_ab, C, A2, A2s


    def _compute_pfos_iteration(self, params, narrowband, pair_covariance, use_tqdm):
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
            use_tqdm (bool): Whether to use a TQDM progress bar.

        Returns:
            rho_abk (numpy.ndarray): The pairwise correlated powers for each frequency.
            sig_abk (numpy.ndarray): The pairwise uncertainties in rho_abk for each frequency.
            Ck (numpy.ndarray): The pair covariance matrix used for each frequency.
            Sk (numpy.ndarray): The PSD(f_k)/T_{span} for each frequency.
            Sks (numpy.ndarray): The uncertainty or covariance matrix in Sk for each frequency.
        """
        X,Z = self._compute_XZ(params)
        gw_signal = [s for s in self.pta._signalcollections[0] if s.signal_id==self.gwb_name][0]
        phi = gw_signal.get_phi(params)
        rho_abk, sig_abk, norm_abk = self._compute_rhok_sigk(X,Z,phi,narrowband)

        Sk = np.zeros( (self.nfreq, self.norfs) )
        Sks = np.zeros( (self.nfreq, self.norfs,self.norfs) )
        if pair_covariance:
            Ck = np.zeros( (self.nfreq,self.npairs,self.npairs) )
        else:
            Ck = np.zeros( (self.nfreq,self.npairs) )

        for k in range(self.nfreq) if not use_tqdm else tqdm(range(self.nfreq)):
            sk = phi[2*k]

            phi1 = np.zeros( shape=(len(self.freqs)) )  
            phi1[k] = 1
            phi1 = np.repeat(phi1,2)

            phi2 = phi / sk

            if pair_covariance:
                solve_method = 'pinv'
                if self.norfs>1:
                    # MCOS
                    Ck[k] = pc._compute_mcos_pair_covariance(Z, phi1, phi2, self._orf_matrix, 
                            self.orf_design_matrix, rho_abk[k], sig_abk[k], norm_abk[k], sk, 
                            False, self._max_chunk)
                else:
                    # Single component
                    Ck[k] = pc._compute_pair_covariance(Z, phi1, phi2, 
                            self._orf_matrix[0], norm_abk[k], sk, False, self._max_chunk)
            else:
                solve_method='diagonal'
                Ck[k] = sig_abk[k]**2
        
            s, ssig = utils.linear_solve(self.orf_design_matrix, Ck[k], 
                                         rho_abk[k,:,None], solve_method)
            
            Sk[k] = np.squeeze(s) if self.norfs>1 else s.item()
            Sks[k] = np.squeeze(ssig) if self.norfs>1 else np.sqrt(ssig.item()) 

        Sk = np.squeeze(Sk) 
        Sks = np.squeeze(Sks)
        Ck = np.squeeze(Ck)

        return rho_abk, sig_abk, Ck, Sk, Sks

    
    def _compute_rho_sig(self, X, Z, phihat):
        """Compute the rho_ab, sigma_ab correlation and uncertainties

        For Internal Use of the OSplusplus. Users are not recommended to use this!

        This method calculates the rho_ab and sigma_ab for each pulsar pair using
        the OS' X, Z and phihat matrix products. Check Appendix A from 
        Pol et al. 2023, or Gersbach et al. 2024 for matrix product definitions.

        Args:
            X (numpy.ndarray): An array of X matrix products for each pulsar.
            Z (numpy.ndarray): An array of Z matrix products for each pulsar.
            phihat (numpy.ndarray): A vector of the diagonal \hat{\phi} matrix.

        Returns:
            numpy.ndarray, numpy.ndarray: The rho_ab and sigma_ab pairwise correlations
        """
        # rho_ab = (X[a] @ phihat @ X[b]) / tr(Z[a] @ phihat @ Z[b] @ phihat)
        # sig_ab = np.sqrt( tr(Z[a] @ phihat @ Z[b] @ phihat) )
        a,b = self._pair_idx[:,0], self._pair_idx[:,1]

        numerator = np.einsum('ij,ij->i',X[a],(phihat*X[b]))
        Zphi = phihat*Z
        denominator = np.einsum('ijk,ikj->i',Zphi[a],Zphi[b])

        rho_ab = numerator/denominator
        sig_ab = 1/np.sqrt(denominator)

        return rho_ab, sig_ab


    def _compute_rhok_sigk(self, X, Z, phi, narrowband):
        """Compute the rho_ab(f_k), sigma_ab(f_k), and normalization_ab(f_k)

        For Internal Use of the OSplusplus. Users are not recommended to use this!

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

        return rho_abk, sig_abk, norms_abk

