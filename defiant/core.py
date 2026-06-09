
# Defiant imports
from defiant import utils
from defiant.jitting import jit, jit_method
from defiant.linear_solver import chi2_solve
from defiant.pta_wrappers import Enterprise_model, get_enterprise_gw_frequencies
from defiant import pair_covariance as pc, orf_functions as orf_funcs
from defiant import custom_exceptions as os_ex

# Non jax
import numpy as np
from tqdm import tqdm
from warnings import warn
from la_forge.core import Core

# Jax imports
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import random as jrandom


class OptimalStatistic:

    # Setup methods-------------------------------------------------------------
    def __init__(self, psrs, pta, gwb_name='gw', # Needed for initialization
                 core=None, chain=None, param_names=None, # For set_chain
                 orfs=['hd'], pcmc_orf='hd', # For set_orf
                 stabilize_z=None): # Extra options

        self.pta = pta # Mostly for users, PTA wrapper does most of the work
        self.gwb_name = gwb_name # Assume that the GWB name is right
        # Create the PTA wrapper (Potentially include discovery at some point?)
        self.pta_wrapper = Enterprise_model(self.pta, self.gwb_name)
        
        if not hasattr(psrs, '__iter__'): # Make sure pulsars are iterable
            raise ValueError("psrs must be an iterable of pulsars.")
        self.psr_names = [p.name for p in psrs]
        self.npsr = len(self.psr_names)
        # Make sure pulsar name order match PTA
        for i in range(self.npsr):
            if self.psr_names[i] != self.pta.pulsars[i]:
                raise ValueError("Pulsar name order does not match PTA!")
        self.psrs = psrs # Duck typing pulsar objects
        self.psr_pos = jnp.array([p.pos for p in self.psrs]) # Pulsar positions
        
        self.npair = self.npsr*(self.npsr-1)//2 # Number of unique pairs
        # BE CAREFUL CHANGING self.pair_idx, it is assumed elsewhere!
        self.pair_idx = jnp.array(jnp.triu_indices(self.npsr,1)).T # Indexes of unique pairs [n_pairs, 2]
        self.pair_names = [(self.psr_names[a], self.psr_names[b]) for a,b in self.pair_idx]

        self.freqs = get_enterprise_gw_frequencies(self.pta, self.gwb_name)
        self.nfreq = len(self.freqs)
        
        # Set the core and chain parameters (if provided)
        self.set_chain(core, chain, param_names)
        
        # Set up ORF attributes
        self.set_orf(orfs, pcmc_orf)

        # Some extra attributes used by various things
        self.previous_output = {} # Used to store the previous OS outputs
        self._pc_case_idx = None # Used for pair covariance. Will be replaced if needed.
        self._warn_narrowband_norm = False # Used to only warn users once

        # Experimental stuff ---------------------------------------------------
        self.stabilize_z = stabilize_z
        if stabilize_z is not None:
            warn("Stabilizing Z matrix products is an experimental feature. Use with caution.")
        

    # Chain and ORF setup methods-----------------------------------------------
    def set_chain(self, core=None, chain=None, param_names=None):
        # Prefered order for loading chains: 
        # core > core (as path) > chain (as path) > chain + param_names > chain
        if core is not None and isinstance(core, Core):
            self.lfcore = core # Easy peasy!
        elif type(core) is str:
            self.lfcore = Core(corepath=core) # Load core from path
        elif chain is not None and type(chain) is str:
            self.lfcore = Core(chaindir=chain) # Load chain from path
        elif chain is not None and param_names is not None:
            # Assume that the chain ordering is the same as the parameter names
            self.lfcore = Core(chain=chain,params=param_names) 
        elif chain is not None:
            # Assume that the chain ordering is the same as the PTA parameter names
            self.lfcore = Core(chain=chain,params=self.pta.param_names)
        else:
            self.lfcore = None # No chain provided, set to None
            return # Nothing supplied!

        # Set the maximum likelihood parameters
        params, idx = utils.get_max_like_params(self.lfcore)
        self.max_like_params = params
        self.max_like_idx = idx


    def set_orf(self, orfs=['hd'], pcmc_orf='hd'):

        # Need to check if orfs is a string, list, or function
        if not hasattr(orfs, '__iter__') or isinstance(orfs, str):
            orfs = [orfs] # Make it a list if it is not
        
        self.norfs = len(orfs) # Number of ORFs
        self.orf_names = [None]*self.norfs # Names of the ORFs
        self.orf_functions = [None]*self.norfs # The ORF functions themselves
        orf_matrix = np.zeros( (self.norfs, self.npsr, self.npsr) ) # Will be jax array
        design_matrix = np.zeros( (self.npair, self.norfs) ) # Will be jax array

        for i,cur_orf in enumerate(orfs): # Hence why we listified orfs
            if isinstance(cur_orf, str): # ORF is a pre-defined function from orf_functions.py
                self.orf_names[i] = cur_orf
                self.orf_functions[i] = orf_funcs.get_orf_function(cur_orf)
            else: # ORF is user supplied function
                self.orf_names[i] = cur_orf.__name__
                self.orf_functions[i] = cur_orf
            
            for j,(a,b) in enumerate(self.pair_idx):
                value = self.orf_functions[i](self.psrs[a], self.psrs[b])
                design_matrix[j,i] = value
                orf_matrix[i,a,b] = value # Symmetric ORF matrix
                orf_matrix[i,b,a] = value # Symmetric ORF matrix
        
        self.orf_matrix = jnp.array(orf_matrix) # [Norfs, Npsr, Npsr]
        self.orf_design_matrix = jnp.array(design_matrix) # [Npair, Norfs]
            
        # Pair correlation + Multi-component causes issues, use a singular ORF
        # matrix as the approximate pair correlation matrix. This isn't exact,
        # but much more numerically stable--------------------------------------

        pcmc_orf_matrix = np.zeros( (self.npsr,self.npsr) )

        if isinstance(pcmc_orf, str):
            self.pcmc_orf_name = pcmc_orf
            self.pcmc_orf_function = orf_funcs.get_orf_function(pcmc_orf)
        else:
            self.pcmc_orf_name = pcmc_orf.__name__
            self.pcmc_orf_function = pcmc_orf
            
        for j,(a,b) in enumerate(self.pair_idx):
            value = self.pcmc_orf_function(self.psrs[a], self.psrs[b])
            pcmc_orf_matrix[a,b] = value
            pcmc_orf_matrix[b,a] = value
            
        self.pcmc_orf_matrix = jnp.array(pcmc_orf_matrix) # [Npsr, Npsr]
    

    # Optimal statistic computation methods-------------------------------------
    def compute_OS(self, params=None, N=1, gamma=None, pair_covariance=False, 
                   return_pair_vals=False, use_tqdm=True):
        # Parse the parameters
        pars, idx, gam = self._parse_params(params, N, gamma, pfos=False)
        n = len(pars) # Number of iterations to run

        # Setup returns (useful for stopping mid-way)
        self.nmos_iter = {}
        self.nmos_iter = {'A2':[], 'Var_A2':[], 'Gamma':[], 'idx':[]} 
        if return_pair_vals:
            xi_ab = utils.compute_pulsar_pair_separations(self.psr_pos, self.pair_idx)
            self.nmos_iter.update({'xi_ab':xi_ab, 'rho_ab':[], 'sig_ab':[], 'C_ab':[]})

        # Setup iterator
        iterable = zip(pars, gam, idx) 
        if n > 1 and use_tqdm:
            iterable = tqdm(iterable, total=n, desc='NMOS iterations')
        
        # NMOS iterations
        for p, g, i in iterable:
            # Compute the X and Z matrices
            X, Z = self.compute_XZ(p) # [npsr, gw_modes], [npsr, gw_modes, gw_modes]
            # Compute phihat 
            phihat = self.get_phihat(g) # [gw_modes]
            # Get the pairwise correlations and uncertainties
            rho_ab, sig_ab = self.compute_rho_sig(X, Z, phihat) # [npair], [npair]

            # Compute the pair covariance matrix
            if pair_covariance:
                method = 'cholesky'
                # Case index for pair covariance may need to be calculated (only once)
                if self._pc_case_idx is None:
                    self._pc_case_idx = pc.calculate_case_idx(self.npsr)

                # Need to get estimate of GW amplitude. We can circumvent this by
                # combining phihat and A^2 to get phi, which we can get from 
                # the PTA wrapper! This may cause problems when the CURN model is
                # not a powerlaw, meaning that phi != A^2 * phihat. The upside
                # is that this is much better representation of the PSD. 
                phi = self.pta_wrapper.get_gw_phi(p) # [gw_modes]

                # PC + MC must use pcmc_orf for stability
                orf = self.orf_matrix[0] if self.norfs == 1 else self.pcmc_orf_matrix # [npsr, npsr]                    
                
                C_ab = pc.calculate_pair_covariance(Z, phihat, phi, orf, sig_ab**2, 
                                                    self._pc_case_idx) # [npair, npair]
            else:
                method = 'diagonal'
                C_ab = sig_ab**2
            
            # Solve the linear system, shapes are: [norfs], [norfs, norfs]
            theta, cov = chi2_solve(self.orf_design_matrix, C_ab, rho_ab, method)

            # Store the results
            self.nmos_iter['A2'].append(theta)
            self.nmos_iter['Var_A2'].append(cov)
            self.nmos_iter['Gamma'].append(g)
            self.nmos_iter['idx'].append(i)
            if return_pair_vals:
                self.nmos_iter['rho_ab'].append(rho_ab)
                self.nmos_iter['sig_ab'].append(sig_ab)
                self.nmos_iter['C_ab'].append(C_ab)
        
        # Done with NMOS loop
        # Convert lists to arrays
        self.nmos_iter['A2'] = jnp.squeeze( jnp.array(self.nmos_iter['A2']) )
        self.nmos_iter['Var_A2'] = jnp.squeeze( jnp.array(self.nmos_iter['Var_A2']) )
        self.nmos_iter['Gamma'] = jnp.squeeze( jnp.array(self.nmos_iter['Gamma']) )
        self.nmos_iter['idx'] = jnp.squeeze( jnp.array(self.nmos_iter['idx']) )
        if return_pair_vals:
            self.nmos_iter['rho_ab'] = jnp.squeeze( jnp.array(self.nmos_iter['rho_ab']) )
            self.nmos_iter['sig_ab'] = jnp.squeeze( jnp.array(self.nmos_iter['sig_ab']) )
            self.nmos_iter['C_ab'] = jnp.squeeze( jnp.array(self.nmos_iter['C_ab']) )

        return self.nmos_iter


    def compute_PFOS(self, params=None, N=1, pair_covariance=False, select_freq=None,
                     narrowband_norm=False, return_pair_vals=False , use_tqdm=True):
        # Warn the user if they are using narrowband normalization. 
        # Its unlikely what you want.
        if narrowband_norm and not self._warn_narrowband_norm:
            msg = "Using the PFOS with narrowband normalization is sub-optimal " +\
                  "and far more biased form of the PFOS. Use this only if you need " +\
                  "frequency independence in the normalization assumption."
            warn(msg)
            self._warn_narrowband_norm = True

        # Parse the parameters
        pars, idx, _ = self._parse_params(params, N, None, pfos=True)
        n = len(pars) # Number of iterations to run

        # Frequencies for which to analyze the PFOS
        k_idx = list(range(self.nfreq)) if select_freq is None else [select_freq]
        nk = len(k_idx) # Number of frequencies to analyze

        # Setup returns (useful for stopping mid-way)
        self.nmos_iter = {}
        self.nmos_iter = {'Sk':[], 'Var_Sk':[], 'idx':[]} 
        if return_pair_vals:
            xi_ab = utils.compute_pulsar_pair_separations(self.psr_pos, self.pair_idx)
            self.nmos_iter.update({'xi_ab':xi_ab, 'rho_abk':[], 
                                   'sig_abk':[], 'C_abk':[]})

        # Setup iterator
        iterable = zip(pars, idx) 
        if n > 1 and use_tqdm:
            iterable = tqdm(iterable, total=n, desc='NMOS iterations')
        
        # NMOS iterations
        for p, i in iterable:
            # Fill in the empty lists
            self.nmos_iter['Sk'].append([None]*nk)
            self.nmos_iter['Var_Sk'].append([None]*nk)
            if return_pair_vals:
                self.nmos_iter['rho_abk'].append([None]*nk)
                self.nmos_iter['sig_abk'].append([None]*nk)
                self.nmos_iter['C_abk'].append([None]*nk)

            # Per-frequency iterable
            f_iterable = enumerate(k_idx)
            if nk > 1 and use_tqdm:
                f_iterable = tqdm(f_iterable, total=nk, desc='PFOS frequencies', leave=False)
            # This makes an iterable where the first element is the index of the
            # frequency in the loop while the second element is the actual frequency
            # index. The former is for storing results and the latter is for 
            # proper computation.
            
            # Some components can be gotten outside the frequency loop
            # Compute the X and Z matrices
            X, Z = self.compute_XZ(p) # [npsr, gw_modes], [npsr, gw_modes, gw_modes]
            # Compute phi 
            phi = self.get_gw_phi(p) # [gw_modes]

            # Per-frequency loop
            for store_idx, k in f_iterable:
                # store_idx is the index for storing results, k is the actual frequency index
                # Shapes are: [npair], [npair], [npair]
                rho_abk, sig_abk, norm_abk = self.compute_rhok_sigk(X, Z, phi, k, narrowband_norm)
                # Compute the pair covariance matrix
                if pair_covariance:
                    method = 'cholesky'
                    # Case index for pair covariance may need to be calculated (only once)
                    if self._pc_case_idx is None:
                        self._pc_case_idx = pc.calculate_case_idx(self.npsr)

                    # Get the frequency selector matrix. All frequencies except k are 0.
                    phitilde = jnp.repeat(jnp.arange(self.nfreq)==k, 2) # [gw_modes]

                    # Need to get estimate of GW amplitude. We can circumvent this by
                    # combining phihat and A^2 to get phi, which we can get from 
                    # the PTA wrapper! This may cause problems when the CURN model is
                    # not a powerlaw, meaning that phi != A^2 * phihat. The upside
                    # is that this is much better representation of the PSD. 
                    phi = self.pta_wrapper.get_gw_phi(p) # [gw_modes]
                    
                    if narrowband_norm:
                        # Zero out all frequencies except k in phi = (Sk * phitilde)
                        phi = phi * phitilde # [gw_modes]

                    # PC + MC must use pcmc_orf for stability
                    orf = self.orf_matrix[0] if self.norfs == 1 else self.pcmc_orf_matrix # [npsr, npsr]

                    C_abk = pc.calculate_pair_covariance(Z, phitilde, phi, orf, norm_abk**2, 
                                                         self._pc_case_idx) # [npair, npair]
                else:
                    method = 'diagonal'
                    C_abk = sig_abk**2

                # Solve the linear system, shapes are: [norfs], [norfs, norfs]
                Sk, Var_Sk = chi2_solve(self.orf_design_matrix, C_abk, rho_abk, method)



                # Store the results (Note that we need to use k_idx for storage!)
                self.nmos_iter['Sk'][-1][store_idx] = Sk
                self.nmos_iter['Var_Sk'][-1][store_idx] = Var_Sk
        
                if return_pair_vals:
                    self.nmos_iter['rho_abk'][-1][store_idx] = rho_abk
                    self.nmos_iter['sig_abk'][-1][store_idx] = sig_abk
                    self.nmos_iter['C_abk'][-1][store_idx] = C_abk

            # Done with frequency loop
            self.nmos_iter['idx'].append(i)
        
        # Done with NMOS loop
        # Convert lists to arrays
        self.nmos_iter['Sk'] = jnp.squeeze( jnp.array(self.nmos_iter['Sk']) )
        self.nmos_iter['Var_Sk'] = jnp.squeeze( jnp.array(self.nmos_iter['Var_Sk']) )
        self.nmos_iter['idx'] = jnp.squeeze( jnp.array(self.nmos_iter['idx']) )
        if return_pair_vals:
            self.nmos_iter['rho_abk'] = jnp.squeeze( jnp.array(self.nmos_iter['rho_abk']) )
            self.nmos_iter['sig_abk'] = jnp.squeeze( jnp.array(self.nmos_iter['sig_abk']) )
            self.nmos_iter['C_abk'] = jnp.squeeze( jnp.array(self.nmos_iter['C_abk']) )

        return self.nmos_iter


    def compute_XZ(self, params):
        # Use the PTA wrapper!
        X, Z = self.pta_wrapper.get_XZ(params)
        if self.stabilize_z is not None:
            Z = utils.stabilize_covariance(Z, self.stabilize_z)
        
        # Make sure X and Z are not NaN or inf
        if utils.is_nan_or_inf(X):
            raise ValueError("X matrix contains NaN or inf values!")
        if utils.is_nan_or_inf(Z):
            raise ValueError("Z matrix contains NaN or inf values!")
        
        return X, Z
    
    
    def get_gw_phi(self, params):
        return self.pta_wrapper.get_gw_phi(params)
    

    def get_phihat(self, gamma):
        return utils.powerlaw_psd(self.freqs, 0.0, gamma, modes=2) # [gw_modes]
    
    
    def compute_rho_sig(self, X, Z, phihat):
        # Just use the jitted helper function!
        rho_ab, sig_ab = _compute_rho_sig(X, Z, phihat)

        # Check for NaN or inf values in rho_ab and sig_ab
        if utils.is_nan_or_inf(sig_ab):
            print(os_ex.NaNPairwiseError.extended_response())
            raise os_ex.NaNPairwiseError('NaN or inf values found in pairwise uncertainties (sig_ab).')
        if utils.is_nan_or_inf(rho_ab):
            raise os_ex.NaNPairwiseError('NaN or inf values found in pairwise correlations (rho_ab).')
        
        return rho_ab, sig_ab
    

    def compute_rhok_sigk(self, X, Z, phi, k, narrowband_norm=False):
        # Just use the jitted helper function!
        rho_abk, sig_abk, norm_abk = _compute_rhok_sigk(X, Z, phi, k, narrowband_norm)

        # Check for NaN or inf values in rho_abk and sig_abk
        if utils.is_nan_or_inf(sig_abk):
            print(os_ex.NaNPairwiseError.extended_response())
            raise os_ex.NaNPairwiseError('NaN or inf values found in pairwise uncertainties (sig_abk).')
        if utils.is_nan_or_inf(rho_abk):
            raise os_ex.NaNPairwiseError('NaN or inf values found in pairwise correlations (rho_abk).')
        # If norm_abk is NaN or inf, so will rho_abk and sig_abk
        
        return rho_abk, sig_abk, norm_abk


    # Helper methods------------------------------------------------------------
    def get_params(self, N=1, idx=None):
        if self.lfcore is None:
            msg = 'No chain parameters set! Set these before calling this method.'
            raise ValueError(msg)
        
        if idx is None:
            # If indexes are not supplied, then choose them randomly without burn-in
            idx = np.random.randint(self.lfcore.burn, self.lfcore.chain.shape[0], N)
        else:
            if isinstance(idx, (int, np.integer)): # Sometimes idx is a single int!
                idx = [idx] # put it in a list so np.array() is happy
            idx = jnp.array(idx) # into an array of indices

        samples = []
        for i in idx: # Not sure if there is a better way than just a for loop
            d = {p:v for p,v in zip(self.lfcore.params, self.lfcore.chain[i])}
            samples.append(d)

        return {'samples': samples, 'idx': idx}
        

    def _parse_params(self, params, N, gamma=None, pfos=False):
        # If params is not None, then we ignore N
        # Get the parameters as a list of dictionaries
        if params is not None:# Something in params!
            if isinstance(params, dict):
                # A single dictionary of parameters
                idx = [-1] # No index associated
                pars = [params] # Put in a list
            elif hasattr(params, '__iter__') and isinstance(params[0], dict):
                # An iterable of dictionaries
                idx = [-1]*len(params) # No indexes associated
                pars = params # Already a list of dictionaries
            elif hasattr(params, '__iter__') and isinstance(params[0], (int, np.integer)):
                # An iterable of indexes
                idx = params # Given a list of indexes
                pars = self.get_params(idx=params)['samples'] # Get the parameters
            elif isinstance(params, (int, np.integer)):
                # A single index
                idx = [params] # given a single index
                pars = self.get_params(idx=params)['samples'] # Get the parameters
            else:
                msg = "params must be a dictionary, list of dictionaries, or list of indexes."
                raise ValueError(msg)
            
        elif N>1: # params is None and N>1
            # Default to noise marginalization
            out = self.get_params(N) # Get N random samples
            pars, idx = out['samples'], out['idx']
        
        elif N==1: # params is None and N==1
            # Default to maximum likelihood
            idx = [self.max_like_idx] # Index of max like
            pars = [self.max_like_params] # Max like parameters

        else:
            msg = "Unable to determine parameters. Set either params or N>=1."
            raise ValueError(msg)
        
        if pfos:
            # If PFOS, we don't need gamma, so just return it!
            return pars, idx, None
        
        
        # Deal with gamma values -----------------------------------------------
        if gamma is not None:
            # Override gamma, check if list or single value
            if hasattr(gamma, '__iter__'):
                gam = gamma # Assume it is an iterable of floats
            elif isinstance(gamma, (float, np.floating)):
                gam = [gamma]*len(pars) # Single float, make list of floats
            else:
                msg = "gamma must be a float or a list of floats."
                raise ValueError(msg)
        
            if len(gam) != len(pars): # Check if gamma is the right length
                msg = "gamma must be a single float or the same length as the number of iterations."
                raise ValueError(msg)
        
        elif self.gwb_name+'_gamma' in pars[0]: 
            # Gamma is none, but is in the parameters!
            gam = [p[self.gwb_name+'_gamma'] for p in pars]

        elif self.pta_wrapper.fixed_gamma is not None:
            # Gamma is none, not in parameters, but the PTA is fixed gamma!
            gam = self.pta_wrapper.fixed_gamma
            gam = [gam]*len(pars) # Make it a list of gammas

        else:
            msg = "Unable to determine gamma values. Set gamma or include in parameters."
            raise ValueError(msg)
        
        return pars, idx, gam


# Jitted helper functions
# Pairwise correlation functions------------------------------------------------
@jit
def _compute_rho_sig(X, Z, phihat):
    # Assumes same pair_index as in OptimalStatistic
    npsr, nm = X.shape # Number of pulsars, number of GW modes
    a,b = jnp.triu_indices(npsr,1) # Get the pulsar indexes for each pair
    
    Zphih = Z*phihat[None,:,None] # Precompute Z @ diag(phihat) [N_psr, gw_mode, gw_mode]

    # Numerator and denominator of the optimal statistic pairwise correlations
    num = jnp.einsum('ni,i,ni->n', X[a], phihat, X[b]) # [N_pair]
    den = jnp.einsum('nij,nji->n', Zphih[a], Zphih[b]) # [N_pair]

    rho_ab = num/den # [N_pair]
    sig_ab = 1/jnp.sqrt(den) # [N_pair]

    return rho_ab, sig_ab


@jit(static_argnums=(4))
def _compute_rhok_sigk(X, Z, phi, k, narrowband_norm):
    # Assumes same pair_index as in OptimalStatistic
    npsr, nm = X.shape # Number of pulsars, number of GW modes
    a,b = jnp.triu_indices(npsr,1) # Get the pulsar indexes for each pair
    Sk = phi[2*k] # Get the power at frequency k (assuming phi is diagonal)

    # Get the frequency selector matrix. All frequencies except k are 0. 
    phit = (jnp.arange(nm//2)==k) * jnp.ones(nm//2) # [gw_freqs]
    phit = jnp.repeat(phit,2) # [gw_modes] Repeat for sine and cosine modes
    # Get frequency normalized phi matrix (big Phi)
    Phi = phi / Sk # [gw_modes]
    # big Phi is replaced with the frequency selector matrix if narrowband_norm
    if narrowband_norm:
        Phi = phit

    Zphit = Z*phit[None,:,None] # Precompute Z @ diag(phit) [N_psr, gw_mode, gw_mode]
    ZPhi = Z*Phi[None,:,None] # Precompute Z @ diag(Phi) [N_psr, gw_mode, gw_mode]


    norm_abk = 1/(jnp.einsum('nij,nji->n',Zphit[a],ZPhi[b]))
    
    rho_abk = jnp.einsum('ni,i,ni->n', X[a], phit, X[b]) * norm_abk
    sig_abk = jnp.einsum('nij,nji->n', Zphit[a], Zphit[b]) * norm_abk**2
    sig_abk = jnp.sqrt(sig_abk)

    return rho_abk, sig_abk, norm_abk

