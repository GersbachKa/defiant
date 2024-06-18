
from OSplusplus.os_Exceptions import *
from OSplusplus.utils import *
from OSplusplus.utils import _gwb_a2_from_freqs
from OSplusplus.pair_covariance_functions import _compute_mcos_pair_covariance, \
    _compute_factored_pair_covariance

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

    def __init__(self, psrs, pta, gwb_name='gw', corepath = None, core = None,  
                 chain_path = None, chain = None, param_names = None, orfs = ['hd']):
        
        # Order of psrs needs to be the same as the one given to pta
        # Check to make sure the type is correct
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
        
        self.lfcore = None
        self.max_likelihood_params = None
        self.set_la_forge_core(corepath, core, chain_path, chain, param_names)

        self.frequencies = get_pta_frequencies(pta,gwb_name)
        
        self.pair_names = [(a,b) for i,a in enumerate(psr_names) for b in psr_names[i+1:]]
        self._pair_index = np.array([(a,b) for a in range(self.npsr) for b in range(a+1,self.npsr)])
        self.npairs = len(self.pair_names)

        self.orf_design_matrix = None
        self._orf_pair_matrix = None
        self.orf_names = None
        self.set_orf(orfs)

        self.nmos_iterations = None


    def set_la_forge_core(self, corepath, core, chain_path, chain, param_names):

        # Prefered order for loading chains: 
        # core > corepath > chain_path > chain + param_names > chain
        if core is not None:
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
            msg = 'No MCMC samples were given! Set these later or supply them when ' +\
                  'computing the OS.'
            warn(msg)

        if self.lfcore is not None:
            self.max_likelihood_params = get_max_likelihood_params(self.lfcore)


    def set_orf(self,orfs=['hd'],names=[None]):

        if not hasattr(orfs, '__getitem__'):
            orfs = [orfs]
        elif type(orfs) == str:
            orfs = [orfs]

        orf_names = []
        orf_des = np.zeros( (len(orfs),self.npairs) ) # Used internal and external
        orf_mat = np.ones( (len(orfs),self.npsr,self.npsr) ) # Used for pair covariance

        for i,orf in enumerate(orfs):
            if type(orf) == str:
                # ORF must be one of the built-in functions
                if orf.lower() == 'hd':
                    cur_orf = model_orfs.hd_orf
                    orf_names.append('HD')
                elif orf.lower() == 'dipole':
                    cur_orf = model_orfs.dipole_orf
                    orf_names.append('Dipole')
                elif orf.lower() == 'monopole':
                    cur_orf = model_orfs.monopole_orf
                    orf_names.append('Monopole')
                elif orf.lower() == 'gw_dipole':
                    cur_orf = model_orfs.gw_dipole_orf
                    orf_names.append('GW_Dipole')
                elif orf.lower() == 'gw_monopole':
                    cur_orf = model_orfs.gw_monopole_orf
                    orf_names.append('GW_Monopole')
                elif orf.lower() == 'st':
                    cur_orf = model_orfs.st_orf
                    # TODO: Check that this is scalar transpose
                    orf_names.append('Scalar transpose')
                else:
                    msg = f"Unknown ORF name: '{orf}'. Check the documentation " +\
                           "for pre-programmed ORFs or supply your own."
                    raise NameError(msg)
                
                for j,(a,b) in enumerate(self._pair_index):
                    v = cur_orf(self.psrs[a].pos , self.psrs[b].pos)
                    orf_des[i,j] = v
                    orf_mat[i,a,b] = v
                    orf_mat[i,b,a] = v

            else:
                # ORF is user supplied function
                orf_names.append(orf.__name__)
                try:
                    for j,(a,b) in enumerate(self._pair_index):
                        v = orf(self.psrs[a], self.psrs[b])
                        orf_des[i,j] = v
                        orf_mat[i,a,b] = v
                        orf_mat[i,b,a] = v
                except Exception as e:
                    msg = f"Cannot use custom ORF function {orf}. Ensure that " +\
                           "the function has two parameters, both of which accept "+\
                           "the 'enterprise.pulsar.BasePulsar' as types."
                    raise TypeError(msg) from e
        
        self.orf_design_matrix = orf_des
        self._orf_pair_matrix = orf_mat # Used for pair covariance calculation
        self.orf_names = orf_names


    def compute_os(self, params=None, N=1, pair_covariant=True, gamma=None, 
                   solve_method=None, use_tqdm=True, max_chunk=300, seed=None):
        if N<1:
            msg = 'Number of noise marginalized iterations n is not valid! Use n>=1'
            raise ValueError(msg)
        
        elif N==1:
            # Single OS (max-likelihood?)
            if params is None and self.lfcore is not None:
                msg = 'No parameters set without noise marginalization, defaulting '+\
                      'to maximum likelihood OS.'
                warn(msg)

                params = self.max_likelihood_params
            else:
                msg = "No parameters given and no chain files to default to!"
                raise ValueError(msg)
            
            pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

            A2,A2s,total_snr = self._compute_os_iteration(pars, gamma, pair_covariant, 
                                            solve_method, use_tqdm, max_chunk)
            return A2,A2s,total_snr
        
        else:
            # Noise marginalized            
            if self.lfcore is None:
                msg = 'Cannot Noise marginalize with a Null La forge core! '
                raise ValueError(msg)
            else:
                # I would parallelize this, but unfortunately I can't pickle an OS++ object
                # due to the pta requirement. If I could pickle a PTA, it would be trivial!
                seed = np.random.randint(0,int(2**63)) if seed is None else seed
                rng = np.random.default_rng(seed) 
                
                self.nmos_iterations = {'A2':[],'A2s':[],'total_snr':[]}
                try:
                    for iter in tqdm(range(N)) if use_tqdm else range(N):
                        rand_i = rng.integers(self.lfcore.burn, len(self.lfcore.chain))
                        params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.params[rand_i])}
                        pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

                        A2,A2s,total_snr = self._compute_os_iteration(pars, gamma, 
                                                    pair_covariant, solve_method, False, max_chunk)
                        
                        self.nmos_iterations['A2'].append(A2)
                        self.nmos_iterations['A2s'].append(A2s)
                        self.nmos_iterations['total_snr'].append(total_snr)

                except Exception as e:
                    msg = 'Stopping NMOS iterations. Calculated values are can be found in OSplusplus.nmos_iterations.'
                    raise NMOSInteruptionError(msg) from e

                finally:
                    A2 = self.nmos_iterations['A2']
                    A2s = self.nmos_iterations['A2s']
                    total_snr = self.nmos_iterations['total_snr']
                    return A2,A2s,total_snr
                

    def compute_pfos(self, params=None, N=1, narrowband=False, pair_covariant=True, 
                     gamma=None, solve_method=None, use_tqdm=True, max_chunk=300, seed=None):
        if N<1:
            msg = 'Number of noise marginalized iterations n is not valid! Use n>=1'
            raise ValueError(msg)
        
        elif N==1:
            # Single OS (max-likelihood?)
            if params is None and self.lfcore is not None:
                msg = 'No parameters set without noise marginalization, defaulting '+\
                      'to maximum likelihood OS.'
                warn(msg)

                params = self.max_likelihood_params
            else:
                msg = "No parameters given and no chain files to default to!"
                raise ValueError(msg)
            
            pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

            Sf,Sfs,total_snr_f = self._compute_pfos_iteration(pars, gamma, pair_covariant, 
                                            narrowband, solve_method, use_tqdm, max_chunk)
            return Sf,Sfs,total_snr_f 
        
        else:
            # Noise marginalized            
            if self.lfcore is None:
                msg = 'Cannot Noise marginalize with a Null La forge core! '
                raise ValueError(msg)
            else:
                # I would parallelize this, but unfortunately I can't pickle an OS++ object
                # due to the pta requirement. If I could pickle a PTA, it would be trivial!
                seed = np.random.randint(0,int(2**63)) if seed is None else seed
                rng = np.random.default_rng(seed) 
                
                self.nmos_iterations = {'A2':[],'A2s':[],'total_snr':[]}
                try:
                    for iter in tqdm(range(N)) if use_tqdm else range(N):
                        rand_i = rng.integers(self.lfcore.burn, len(self.lfcore.chain))
                        params = {p:v for p,v in zip(self.lfcore.params,self.lfcore.params[rand_i])}
                        pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

                        Sf,Sfs,total_snr_f  = self._compute_pfos_iteration(pars, gamma, pair_covariant, 
                                                        narrowband, solve_method, False, max_chunk)
                        
                        self.nmos_iterations['Sf'].append(Sf)
                        self.nmos_iterations['Sfs'].append(Sfs)
                        self.nmos_iterations['total_snr_f'].append(total_snr_f)

                except Exception as e:
                    msg = 'Stopping NMOS iterations. Calculated values are can be found in OSplusplus.nmos_iterations.'
                    raise NMOSInteruptionError(msg) from e


                Sf = self.nmos_iterations['Sf']
                Sfs = self.nmos_iterations['Sfs']
                total_snr_f = self.nmos_iterations['total_snr_f']
                return Sf,Sfs,total_snr_f


    def compute_cross_correlations(self, params=None, gamma=None, max_chunk=300):
        
        if params is None and self.lfcore is not None:
            msg = 'No parameters set without noise marginalization, defaulting '+\
                  'to maximum likelihood OS.'
            warn(msg)

            params = self.max_likelihood_params
        else:
            msg = "No parameters given and no chain files to default to!"
            raise ValueError(msg)
            
        pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

        X,Z = self._get_XZ(params)

        gamma = get_gwb_gamma(self.pta, params, self.gwb_name) if gamma is None else gamma
        phihat = powerlaw( np.repeat(self.frequencies,2), 0, gamma)

        xi_ab = compute_pulsar_separations(self.psrs, self._pair_index)
        
        

        rho_ab, sig_ab = self._compute_correlations(X, Z, phihat)

        return xi_ab, rho_ab, sig_ab


    def compute_binned_cross_correlations(self,):
        pass


    def compute_per_frequency_cross_correlations(self, params=None, narrowband=False, gamma=None, solve_method=None, use_tqdm=True, max_chunk=300, seed=None):
        if params is None and self.lfcore is not None:
            msg = 'No parameters set without noise marginalization, defaulting '+\
                  'to maximum likelihood OS.'
            warn(msg)

            params = self.max_likelihood_params
        else:
            msg = "No parameters given and no chain files to default to!"
            raise ValueError(msg)
            
        pars = check_pta_params(self.pta, params, len(self.frequencies), self.gwb_name)

        X,Z = self._get_XZ(params)

        gamma = get_gwb_gamma(self.pta, params, self.gwb_name) if gamma is None else gamma
        phihat = powerlaw( np.repeat(self.frequencies,2), 0, gamma)

        xi_ab = compute_pulsar_separations(self.psrs, self._pair_index)
        rho_ab, sig_ab = self._compute_correlations(X, Z, phihat)

        return xi_ab, rho_ab, sig_ab


    def compute_binned_per_frequency_cross_correlations(self):
        pass
    

    def compute_spectral_snrs(self, params=None):
        pass


    def uncertainty_sample(self):
        pass


    def sky_scramble(self,):
        pass


    def phase_shift(self,):
        pass


    def generalized_chisquare(self,):
        pass


    def _get_XZ(self,params):
        """A function to quickly calculate the OS' matrix quantities

        This function calculates the X and Z matrix quantities from the appendix A
        of Pol, Taylor, Romano, 2022: (https://arxiv.org/abs/2206.09936). X and Z
        can be represented as X = F^T @ P^{-1} @ r and Z = F^T @ P^{-1} @ F.

        Args:
            params (dict): A dictionary containing the parameter name:value pairs for the PTA

        Returns:
            (np.array, np.array): A tuple of X and Z. X is an array of vectors for each pulsar 
                (N_pulsar x 2N_frequency). Z is an array of matrices for each pulsar 
                (N_pulsar x 2N_frequency x 2N_frequency)
        """

        nfreq = len(self.frequencies)
        X = np.zeros( shape = ( self.npsr, 2*nfreq ) ) # An array of vectors
        Z = np.zeros( shape = ( self.npsr, 2*nfreq, 2*nfreq ) ) # An array of matrices

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


    def _compute_correlations(self, X, Z, phihat):
        # Compute rho_ab, and sig_ab
        # rho_ab = (X[a] @ phihat @ X[b]) / tr(Z[a] @ phihat @ Z[b] @ phihat)
        # sig_ab = np.sqrt( tr(Z[a] @ phihat @ Z[b] @ phihat) )

        a,b = self._pair_index[:,0], self._pair_index[:,1]

        numerator = (X[a]*X[b]) @ phihat 
        Zphi = Z @ np.diag(phihat)
        denominator = np.einsum('ijk,ikj->i',Zphi[a],Zphi[b])

        rho_ab = numerator/denominator
        sig_ab = 1/np.sqrt(denominator)

        return rho_ab,sig_ab

    
    def _compute_per_frequency_correlations(self, X, Z, phi, narrowband):
        # Compute rho_ab(f_k) for all f_k
        a,b = self._pair_index[:,0], self._pair_index[:,1]

        nf = len(self.frequencies) # Number of frequencies is half the number of elements of X
        npair = len(a)
        rho_abk = np.zeros( (nf,npair) )
        sig_abk = np.zeros( (nf,npair) )
        norms_abk = np.zeros( (nf,npair) )

        for k,sf in enumerate(phi[::2]):
            phi_til = np.zeros( shape=(nf) )  
            phi_til[k] = 1
            phi_til = np.repeat(phi_til,2)

            Phi = phi/sf

            numerator = np.sum(X[a] * phi_til * X[b],axis=1)
            # Don't ask me why this is faster, I don't know, it just is.
            if narrowband:
                denominator = np.einsum('ijk,ikj->i', phi_til*Z[a], phi_til*Z[b]) 
            else:
                denominator = np.einsum('ijk,ikj->i', phi_til*Z[a], Phi*Z[b])

            rho_abk[k] = numerator/denominator
            sig_abk[k] = 1/np.sqrt(denominator)
            norms_abk[k] = 1/denominator

        return rho_abk,sig_abk,norms_abk


    def _compute_os_iteration(self, params, gamma, pair_covariant, solve_method, use_tqdm, max_chunk):

        X,Z = self._get_XZ(params)

        gamma = get_gwb_gamma(self.pta, params, self.gwb_name) if gamma is None else gamma
        phihat = powerlaw( np.repeat(self.frequencies,2), 0, gamma)

        # To do linear chi-square minimization, we need data, model, and covariance
        # model is the orf_matrix which has already been computed
        # Get data (rho_ab) and part of covariance (sig_ab)
        rho_ab, sig_ab = self._compute_correlations(X, Z, phihat)

        # Get covariance and solve
        if pair_covariant:
            if solve_method is not None and solve_method.lower() == 'exact':
                msg = 'It is not reccomended to use exact matrix inverse with pair covariance. '+\
                      'Large condition numbers with large matrices do not play well with inv()'
                warn(msg)
            
            if self.orf_design_matrix.shape[0]>1:
                # Multiple component - From Sardesai et al. 2023
                # Need to compute the a no-PC MCOS for amp estimates
                temp_C = np.diag(np.square(sig_ab))
                mcos_amp_est,_,_ = linear_solve(self.orf_design_matrix.T, temp_C,
                                                rho_ab[:,None], solve_method)
                mcos_amp_est /= np.sum(mcos_amp_est) # Normalize (i.e. power per process)
                a2_est = _gwb_a2_from_freqs(params, gamma, self.frequencies, 
                                            self.gwb_name, self.lfcore)

                mcos_amp_est *= a2_est # Multiply by CURN amplitude

                C = _compute_mcos_pair_covariance(Z, phihat, phihat, self._orf_pair_matrix,
                               sig_ab, self._pair_index, mcos_amp_est, use_tqdm, max_chunk)
            else:
                # Single component
                factored_C = _compute_factored_pair_covariance(Z, phihat, phihat,
                                self._orf_pair_matrix[0], sig_ab, self._pair_index, 
                                use_tqdm, max_chunk)
                
                a2_est = _gwb_a2_from_freqs(params, gamma, self.frequencies, 
                                            self.gwb_name, self.lfcore)

                C = 1*factored_C[0] + a2_est*factored_C[1] + a2_est**2*factored_C[2]
        else:
            C = np.diag(np.square(sig_ab))
        
        A2, A2s, total_snr = linear_solve(self.orf_design_matrix.T, C, rho_ab[:,None], solve_method)
        
        A2 = np.squeeze(A2) if A2.size>1 else A2.item()
        A2s = np.squeeze(A2s) if A2s.size>1 else np.sqrt(A2s.item())
        return A2, A2s, total_snr     
    

    def _compute_pfos_iteration(self, params, gamma, pair_covariant, narrowband, solve_method, use_tqdm, max_chunk):

        X,Z = self._get_XZ(params)

        gamma = get_gwb_gamma(self.pta, params, self.gwb_name) if gamma is None else gamma
        phi = self.pta[0][self.gwb_name].get_phi(params)

        # To do linear chi-square minimization, we need data, model, and covariance
        # model is the orf_matrix which has already been computed
        # Get data (rho_ab(f_k)) and part of covariance (sig_ab(f_k))
        rho_abk, sig_abk, norms_abk = self._compute_per_frequency_correlations(X, Z, phi, narrowband)


        Sf = []
        Sfs = []
        total_snr_k = []

        # Get covariance and solve
        for k in tqdm(range(len(self.frequencies))) if use_tqdm else range(len(self.frequencies)):
            phi1 = np.zeros( shape=(len(self.frequencies)) )  
            phi1[k] = 1
            phi1 = np.repeat(phi1,2)

            phi2 = phi1 if narrowband else phi/phi[2*k]
            if pair_covariant:
                if solve_method is not None and solve_method.lower() == 'exact':
                    msg = 'It is not reccomended to use exact matrix inverse with pair covariance. '+\
                      'Large condition numbers with large matrices do not play well with inv()'
                    warn(msg)
            
                if self.orf_design_matrix.shape[0]>1:
                    # Multiple component - From Sardesai et al. 2023
                    # Need to compute the a no-PC MCOS for amp estimates
                    temp_C = np.diag(np.square(sig_abk[k]))
                    mcos_amp_est,_,_ = linear_solve(self.orf_design_matrix.T, temp_C,
                                                    rho_abk[k,:,None], solve_method)
                    mcos_amp_est /= np.sum(mcos_amp_est) # Normalize (i.e. power per process)
                    a2_est = _gwb_a2_from_freqs(params, gamma, self.frequencies, 
                                                self.gwb_name, self.lfcore)

                    mcos_amp_est *= a2_est # Multiply by CURN amplitude

                    C = _compute_mcos_pair_covariance(Z, phi1, phi2, self._orf_pair_matrix,
                               norms_abk[k], self._pair_index, mcos_amp_est, False, max_chunk)
                else:
                    # Single component
                    factored_C = _compute_factored_pair_covariance(Z, phi1, phi2,
                                    self._orf_pair_matrix[0], norms_abk[k], self._pair_index, 
                                    False, max_chunk)
                
                    a2_est = _gwb_a2_from_freqs(params, gamma, self.frequencies, 
                                                self.gwb_name, self.lfcore)

                    C = 1*factored_C[0] + a2_est*factored_C[1] + a2_est**2*factored_C[2]
            else:
                C = np.diag(np.square(sig_abk[k]))
        
            s, ss, snr_f = linear_solve(self.orf_design_matrix.T, C, rho_abk[k,:,None], solve_method)
            Sf.append(s)
            Sfs.append(ss)
            total_snr_k.append(snr_f)
        
        Sf = np.squeeze(Sf)
        Sfs = np.squeeze(Sfs) if len(np.array(Sfs).shape)>1 else np.sqrt(Sfs)
        total_snr_k = np.array(total_snr_k)
        return Sf, Sfs, total_snr_k

    
    
