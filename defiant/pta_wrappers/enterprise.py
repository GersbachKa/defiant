
from defiant.custom_exceptions import ModelPTAError
from defiant.pta_wrappers.caching import set_cache_params, product_cache
from defiant.utils import is_diagonal

# Non-jax friendly
import numpy as np
from tqdm import tqdm

# Jax imports
import jax.numpy as jnp


class Enterprise_model:
    """A wrapper class to interface with enterprise PTA objects for use in the OS.

    This class is designed to encapsulate and compute the necessary matrix product
    for the optimal statistic using enterprise PTA objects. For the most part,
    the only methods the OS needs from this class are `get_gw_phi` and `get_XZ`.
    The other methods are helpers for these main methods. 

    Note that both `get_gw_phi` and `get_XZ` return jax arrays while the other
    methods return numpy arrays or other. This keeps all of the numpy and
    enterprise specific code in here while the core can stay jax friendly.

    Attributes
    ----------
    pta : enterprise PTA object
        The enterprise PTA object.
    gwb_name : str
        The name of the GWB signal in the PTA.
    psr_signals : list
        A list of the signal collections for each pulsar in the PTA.
    npsr : int
        The number of pulsars in the PTA.
    gw_signal : enterprise signal object
        The enterprise signal object for the GWB signal in the PTA.
    marginalizing_timing_model : bool
        A boolean indicating whether the PTA is using a marginalizing timing model.
    fixed_gamma : float or None
        The fixed gamma value for the GWB signal if it is a fixed slope power-law, otherwise None.
    gw_freqs : array
        The frequencies for the GWB signal in the PTA.
    nfreq : int
        The number of frequencies for the GWB signal in the PTA.
    gw_modes : int
        The number of modes for the F matrix for the GWB signal in the PTA.
    rn_modes : array
        An array containing the number of modes for the T matrix for each pulsar in the PTA.
    same_rn_modes : bool
        A boolean indicating whether all pulsars in the PTA have the same number of T modes.
    freespec_gwb : bool
        A boolean indicating whether the GWB signal in the PTA has free spectrum components.
    freespec_keys : list or None
        A list of the parameter keys for the free spectrum components of the GWB signal if it
        has them, otherwise None.
    """

    def __init__(self, pta, gwb_name):
        """A wrapper class to interface with enterprise PTA objects for use in the OS.

        This class is designed to encapsulate and compute the necessary matrix product
        for the optimal statistic using enterprise PTA objects. For the most part,
        the only methods the OS needs from this class are `get_gw_phi` and `get_XZ`.
        The other methods are helpers for these main methods. 

        Note that both `get_gw_phi` and `get_XZ` return jax arrays while the other
        methods return numpy arrays or other. This keeps all of the numpy and
        enterprise specific code in here while the core can stay jax friendly. 

        Parameters
        ----------
        pta : enterprise PTA object
            The enterprise PTA object.
        gwb_name : str
            The name of the GWB signal in the PTA.

        Raises
        ------
        ValueError
            If the GWB signal is not found in the PTA signals or if not all 
            freespec keys are found in the GWB signal parameters.
        """
        # Use duck typing for the PTA object
        self.pta = pta # The PTA object
        self.gwb_name = gwb_name # Name of the GWB signal in the PTA

        self.psr_signals = self.pta._signalcollections # I don't like typing this a million times
        self.npsr = len(self.psr_signals)

        # Lets also get the GW signal to calculate the GWB phi. 
        # NOTE: Assumes that the spectrum is identical for all pulsars.
        for s in self.psr_signals[0].signals:
            if s.signal_id == self.gwb_name:
                self.gw_signal = s
                break
        if self.gw_signal is None:
            raise ValueError(f'GWB signal with name {self.gwb_name} not found in PTA signals!')[0]

        # Marginalizing timing model check 
        # NOTE: Assumes that all pulsars have the same timing model setup
        self.marginalizing_timing_model = False
        for s in self.psr_signals[0].signals:
            if 'marginalizing linear timing model' in s.signal_name:
                self.marginalizing_timing_model = True
                break

        # Check if gamma is fixed for the GWB signal (e.g. fixed slope power-law)
        try:
            self.fixed_gamma = self.gw_signal.get(self.gwb_name+'_gamma')
        except KeyError:
            self.fixed_gamma = None

        # Get PTA GWB frequencies
        self.gw_freqs = get_enterprise_gw_frequencies(self.pta, self.gwb_name) # [nfreqs]
        self.nfreq = len(self.gw_freqs)
        
        # Get the number of modes for the F matrix
        self.gw_modes = 2*self.nfreq # Same for all pulsars
        # T could have different number of modes
        self.rn_modes = np.zeros(self.npsr, dtype=int)
        for i in range(self.npsr):
            self.rn_modes[i] = self.psr_signals[i].get_basis().shape[1] 
        # Check if all rn_modes are the same for simplicity
        self.same_rn_modes = np.all(self.rn_modes == self.rn_modes[0])

        # Check if this PTA has free spectrum components
        self.freespec_gwb = False
        self.freespec_keys = None
        if f'{self.gwb_name}_log10_rho_0' in self.gw_signal.param_names:
            self.freespec_gwb = True
            self.freespec_keys = [f'{self.gwb_name}_log10_rho_{i}' for i in range(self.nfreq)]
            # Check if the keys exist, just in case
            if not all(k in self.gw_signal.param_names for k in self.freespec_keys):
                raise ValueError('Freespec keys not found in GWB signal parameters!')

        # Set cache parameters (Sets npsr and _cache_params attributes)
        set_cache_params(self, self.pta)

        # Pre-cache all products if desired
        self._precache_matrices()
        

    def _precache_matrices(self):
        """A method to pre-cache all cache-able matrix products for all pulsars.

        This method loops through each pulsar and pre-caches all matrix products
        that can be cached. A matrix product can be cached if it does not depend on
        any variable parameters.
        """
        for i in tqdm(range(self.npsr), desc='Pre-caching matrix products'):
            # Check if this pulsar has model dependent parameters
            has_wn = len(self._cache_params['white_noise'][i]) > 0
            has_basis = len(self._cache_params['basis'][i]) > 0
            has_delay = len(self._cache_params['delay'][i]) > 0

            # Must not have white noise or basis parameters
            if (not has_wn) and (not has_basis):
                # Can cache things!
                _ = self.get_psr_FNF(params={}, idx=i)
                _ = self.get_psr_TNT(params={}, idx=i)
                _ = self.get_psr_TNF(params={}, idx=i)

                if not has_delay:
                    # Can cache dt products as well
                    _ = self.get_psr_FNdt(params={}, idx=i)
                    _ = self.get_psr_TNdt(params={}, idx=i)


    def clean_params(self, params):
        """Clean the input parameter dictionary to work with enterprise

        This method is necessary to handle an enterprise model with a free spectrum
        GWB. For whatever reason, the output chain of a free spectrum GWB often
        separates the GWB parameters into individual log10_rho_i parameters, 
        but the enterprise signal expects a single log10_rho parameter as an array 
        of values.

        Parameters
        ----------
        params : dict
            a dictionary of parameters.

        Returns
        -------
        dict
            a cleaned dictionary of parameters.
        """
        if not self.freespec_gwb:
            # No need to clean params
            return params
        
        # Check if the freespec keys are in params
        if not all(k in params for k in self.freespec_keys):
            # No need to clean params
            return params

        # If we made it here, we need to clean the params
        pars = {}
        pars.update(params) # Start with existing params
        key = f'{self.gwb_name}_log10_rho'
        vals = np.array([params[k] for k in self.freespec_keys])
        pars.update( {key:vals} ) # Add the new parameter key
        return pars


    # Quantities wanted by the OS-----------------------------------------------
    def get_gw_phi(self, params):
        """Get the phi matrix for the GWB signal

        This method gets the phi matrix for the GWB signal from the 
        enterprise signal object. This represents the PSD of the GWB signal
        in the Fourier basis. Returned in units of [s^2].

        NOTE: This method assumes that the gw phi is always diagonal
        NOTE: This method is returned as a jax array.

        Parameters
        ----------
        params : dict
             The dictionary of model parameters

        Returns
        -------
        array
            The phi matrix for the GWB signal [gw_modes]
        """
        pars = self.clean_params(params)

        phi = self.gw_signal.get_phi(pars)

        # Throw an error if phi is more than 1D
        if phi.ndim != 1:
            raise ValueError(f'GWB phi is expected to be 1D (diagonal), but got shape {phi.shape}')
        
        phi = jnp.array(phi) # Ensure jax array
        return phi # [gw_modes]
    
    
    def get_XZ(self, params):
        """Get the X and Z matrices for the whole PTA

        This method gets the X and Z matrices for the whole PTA. These are:
        - X = F^T N^{-1} dt - F^T N^{-1} T (T^T N^{-1} T + phiinv)^-1 T^T N^{-1} dt 
        - Z = F^T N^{-1} F - F^T N^{-1} T (T^T N^{-1} T + phiinv)^-1 T^T N^{-1} F

        X effectively represent a weird weighted projection of the residuals into
        the Fourier domain, while Z represents a weird weighted projection of the 
        Fourier design matrix for the GWB signal. (Be careful with these interpretations!
        as the weighting is non-trivial).

        X has shape (npsr, gw_modes) and Z has shape (npsr, gw_modes, gw_modes).

        NOTE: This method is returned as jax arrays.

        Parameters
        ----------
        params : dict
             The dictionary of model parameters

        Returns
        -------
        tuple
            A tuple containing:
            - X: The X matrix for the whole PTA [npsr, gw_modes]
            - Z: The Z matrix for the whole PTA [npsr, gw_modes, gw_modes]
        """
        pars = self.clean_params(params)

        X = np.zeros((self.npsr, self.gw_modes))
        Z = np.zeros((self.npsr, self.gw_modes, self.gw_modes))
        for i in range(self.npsr):
            FNdt = self.get_psr_FNdt(pars, i) # [gw_modes] 
            TNdt = self.get_psr_TNdt(pars, i) # [rn_modes]
            FNF = self.get_psr_FNF(pars, i) # [gw_modes, gw_modes]
            TNF = self.get_psr_TNF(pars, i) # [rn_modes, gw_modes]
            TNT = self.get_psr_TNT(pars, i) # [rn_modes, rn_modes]
            phiinv = self.get_psr_phiinv(pars, i) # [rn_modes, rn_modes]

            sigma = TNT + phiinv # [rn_modes, rn_modes]

            FNT_Sigma_inv = np.linalg.solve(sigma, TNF).T # [gw_modes, rn_modes]

            # X = F^T N^{-1} dt - F^T N^{-1} T (T^T N^{-1} T + phiinv)^-1 T^T N^{-1} dt
            X[i] = FNdt - FNT_Sigma_inv @ TNdt # [gw_modes]
            # Z = F^T N^{-1} F - F^T N^{-1} T (T^T N^{-1} T + phiinv)^-1 T^T N^{-1} F
            Z[i] = FNF - FNT_Sigma_inv @ TNF # [gw_modes, gw_modes]

        X = jnp.array(X) # Ensure jax array
        Z = jnp.array(Z) # Ensure jax array
        return X, Z

    
    # Non-caching individual pulsar matrix get----------------------------------
    def get_psr_dt(self, clean_params, idx):
        """Get the residual vector for pulsar `idx`

        This method gets the residual vector for pulsar `idx` from the enterprise 
        signal object. Returned in units of [s] and of length Ntoa.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
             The index of the pulsar to get the residuals for

        Returns
        -------
        array
            The residual vector for pulsar `idx` [Ntoa]
        """
        return self.psr_signals[idx].get_detres(clean_params) # [Ntoa]
    

    def get_psr_N(self, clean_params, idx):
        """Get the TOA covariance matrix object for pulsar `idx`

        This method gets the TOA covariance matrix object for pulsar `idx` from the
        enterprise signal object. This is an object that can be used to compute
        matrix products. Note that this does not return the actual covariance 
        matrix, but an object that can be used to compute their products. 
        (see solve_xNy method)

        NOTE: This method assumes you have already cleaned `params` with `clean_params`

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the covariance matrix for

        Returns
        -------
        object
            The covariance matrix object for pulsar `idx`
        """
        return self.psr_signals[idx].get_ndiag(clean_params) # [Ntoa, Ntoa]
    

    def get_psr_F(self, clean_params, idx):
        """Get the Fourier design matrix for pulsar `idx`

        This method gets the Fourier design matrix for pulsar `idx` from the
        enterprise signal object. This is the Fourier basis that the GWB signal is
        projected onto. Returned in units of [s] and of shape (Ntoa, gw_modes).

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the Fourier design matrix for

        Returns
        -------
        array
            The Fourier design matrix for pulsar `idx` [Ntoa, gw_modes]
        """
        return self.psr_signals[idx][self.gwb_name].get_basis(clean_params) # [Ntoa, gw_modes]
    

    def get_psr_T(self, clean_params, idx):
        """Get the full fourier design matrix for pulsar `idx`

        This method gets the full Fourier design matrix for pulsar `idx` from the
        enterprise signal object. This is like the F matrix, but includes all
        fourier components from all signals, not just the GWB signal. Returned in 
        units of [s] and of shape (Ntoa, rn_modes).

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the full Fourier design matrix for

        Returns
        -------
        array
            The full Fourier design matrix for pulsar `idx` [Ntoa, rn_modes]
        """
        return self.psr_signals[idx].get_basis(clean_params) # [Ntoa, rn_modes]
    

    def get_psr_phiinv(self, clean_params, idx):
        """Get the inverse phi matrix for pulsar `idx`

        This method gets the inverse phi matrix for pulsar `idx` from the enterprise
        signal object. This is the inverse of the PSD of the red noise 
        processes for pulsar `idx` in the Fourier basis. Returned in units of [s^-2] 
        and of shape (rn_modes, rn_modes).

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the inverse phi matrix for

        Returns
        -------
        array
            The inverse phi matrix for pulsar `idx` [rn_modes, rn_modes]
        """
        phiinv = self.psr_signals[idx].get_phiinv(clean_params)
        if phiinv.ndim == 1:
            phiinv = np.diag(phiinv) # Ensure 2D array
        return phiinv # [rn_modes, rn_modes]
    

    # Cached pulsar-specific matrix products------------------------------------
    @product_cache(['white_noise', 'basis', 'delay'])
    def get_psr_FNdt(self, clean_params, idx):  
        """Get the product F^T N dt for pulsar `idx`

        This method gets the matrix product F^T N dt for pulsar `idx`. This product
        acts as a whitened representation of the residuals projected into the 
        Fourier domain. This method is cached such that repeated calls which do
        not change the white noise, basis, or delay parameters will be fast.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the inverse phi matrix for

        Returns
        -------
        array
            The product F^T N dt for pulsar `idx` [gw_modes]
        """     
        F = self.get_psr_F(clean_params, idx) # [Ntoa, gw_modes]
        N = self.get_psr_N(clean_params, idx) # [Ntoa, Ntoa]
        dt = self.get_psr_dt(clean_params, idx) # [Ntoa, 1]
        FNdt = self.solve_xNy(N, F, dt) # [gw_modes]
        return FNdt # [npsr, gw_modes]


    @product_cache(['white_noise', 'basis', 'delay'])
    def get_psr_TNdt(self, clean_params, idx):
        """Get the product T^T N dt for pulsar `idx`

        This method gets the matrix product T^T N dt for pulsar `idx`. This product
        acts as a whitened representation of the residuals projected into the
        full Fourier domain (including all red noise processes). This method is
        cached such that repeated calls which do not change the white noise, basis, 
        or delay parameters will be fast.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the product for

        Returns
        -------
        array
            The product T^T N dt for pulsar `idx` [rn_modes]
        """
        T = self.get_psr_T(clean_params, idx) # [Ntoa, rn_modes]
        N = self.get_psr_N(clean_params, idx) # [Ntoa, Ntoa]
        dt = self.get_psr_dt(clean_params, idx) # [Ntoa, 1]
        TNdt = self.solve_xNy(N, T, dt) # [rn_modes]
        return TNdt # [rn_modes]


    @product_cache(['white_noise', 'basis'])
    def get_psr_FNF(self, clean_params, idx):
        """Get the product F^T N F for pulsar `idx`

        This method gets the matrix product F^T N F for pulsar `idx`. This product
        acts as a TOA covariance matrix projected into the Fourier domain. This 
        method is cached such that repeated calls which do not change the white noise 
        or basis parameters will be fast.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the product for

        Returns
        -------
        array
            The product F^T N F for pulsar `idx` [gw_modes, gw_modes]
        """
        F = self.get_psr_F(clean_params, idx) # [Ntoa, gw_modes]
        N = self.get_psr_N(clean_params, idx) # [Ntoa, Ntoa]
        FNF = self.solve_xNy(N, F, F) # [gw_modes, gw_modes]
        return FNF


    @product_cache(['white_noise', 'basis'])
    def get_psr_TNT(self, clean_params, idx):
        """Get the product T^T N T for pulsar `idx`

        This method gets the matrix product T^T N T for pulsar `idx`. This product
        acts as a TOA covariance matrix projected into the full Fourier domain (including
        all red noise processes). This method is cached such that repeated calls 
        which do not change the white noise or basis parameters will be fast.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the product for

        Returns
        -------
        array
            The product T^T N T for pulsar `idx` [rn_modes, rn_modes]
        """
        T = self.get_psr_T(clean_params, idx) # [Ntoa, rn_modes]
        N = self.get_psr_N(clean_params, idx) # [Ntoa, Ntoa]
        TNT = self.solve_xNy(N, T, T) # [rn_modes, rn_modes]
        return TNT


    @product_cache(['white_noise', 'basis'])
    def get_psr_TNF(self, clean_params, idx):
        """Get the product T^T N F for pulsar `idx`

        This method gets the matrix product F^T N T for pulsar `idx`. This product
        acts as a cross-covariance matrix between the Fourier design matrix for the 
        GWB signal and the full Fourier design matrix for all red noise processes. 
        This method is cached such that repeated calls which do not change the white 
        noise or basis parameters will be fast.

        NOTE: This method assumes you have already cleaned `params` with `clean_params`
        NOTE: This method is returned as a numpy array.

        Parameters
        ----------
        clean_params : dict
            The cleaned dictionary of model parameters
        idx : int
            The index of the pulsar to get the product for

        Returns
        -------
        array
            The product F^T N T for pulsar `idx` [gw_modes, rn_modes]
        """
        T = self.get_psr_T(clean_params, idx) # [Ntoa, rn_modes]
        N = self.get_psr_N(clean_params, idx) # [Ntoa, Ntoa]
        F = self.get_psr_F(clean_params, idx) # [Ntoa, gw_modes]
        TNF = self.solve_xNy(N, T, F) # [rn_modes, gw_modes]
        return TNF
    

    # Non-cached matrix product
    def solve_xNy(self, N, x, y):
        """Compute the product x^T N y using the enterprise covariance matrix object

        This method computes the product x^T N y using the enterprise covariance 
        matrix object `N`. This method is necessary because the marginalizing timing 
        model in enterprise needs special handling.
        Note that `x` must be of shape (Ntoa, k) and `y` must be of shape (Ntoa, m) 
        for some k and m.

        NOTE: This method is returned as a numpy array.
 
        Parameters
        ----------
        N : object
             The enterprise covariance matrix object, which can be used to compute products
        x : array
            The left array for the product [Ntoa, k]
        y : array
            The right array for the product [Ntoa, m]

        Returns
        -------
        array
            The product x^T N y [k, m]
        """
        if self.marginalizing_timing_model:
            # The marginalizing timing model doesn't support different L and R :(
            term1 = N.Nmat.solve(y, left_array=x)
            term2 = N.MNF(x).T @ N.MNMMNF(y)
            return term1 - term2
        else:
            return N.solve(y, x)



def get_enterprise_gw_frequencies(pta, gwb_name='gw'):
    """Get the frequencies of the PTA from an enterprise PTA object

    Given an enterprise PTA object, this function extracts the frequencies of the
    GWB signal in units of Hz. This function uses a hack-y method and may break
    in the future if enterprise changes their internal structure.

    Parameters
    ----------
    pta : enterprise PTA object
         An enterprise PTA object.
    gwb_name : str
         The name of the GWB signal in the enterprise PTA object, by default 'gw'.

    Returns
    -------
    array
         An array of the frequencies of the GWB signal in Hz. [nfreqs]

    Raises
    ------
    ModelPTAError
        If the frequencies cannot be extracted from the PTA object.
    """
    try:
        gwb_sig = [s for s in pta._signalcollections[0].signals if s.signal_id==gwb_name][0]
        gwb_sig.get_basis()

        if isinstance(gwb_sig._labels, dict):
            freqs = gwb_sig._labels['']
        else: 
            freqs = gwb_sig._labels

        return freqs[::2]  # Return only sine frequencies
    
    except Exception as e:
        raise ModelPTAError('Unable to get PTA frequencies!') from e


