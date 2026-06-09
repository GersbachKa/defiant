
import numpy as np # only used to find dimensions of parameters
from functools import wraps

def product_cache(param_types):
    """A decorator to cache results that depend on given parameter types and pulsars.

    This decorator is designed to be used on methods of the form `foo(self, params, idx)` 
    where `params` is a dictionary of parameter values and `idx` is the index of 
    the pulsar. The caching mechanism will store the results of the function based 
    on the values of the specified parameters and is cached for each individual pulsar.

    The cache is stored in an attribute named `_cache_{func.__name__}` which is a list of
    pairs of (key, value) for each pulsar. The key is a tuple of the values of the 
    specified parameters, and the value is the result of the function. Note that
    this caching method only caching a single result per pulsar.

    Note that the class using this decorator must have an attribute `_cache_params` 
    and `_npsr` which can be initialized using the `set_cache_params` function.

    Parameters
    ----------
    param_types : str or list of str
        The type(s) of parameters to cache on. This should be one or more of
        'white_noise', 'basis', or 'delay'.

    Returns
    -------
    function
        The decorated function with caching functionality.

    Raises
    ------
    ValueError
        If the class does not have the `_cache_params` attribute initialized.
    """
    # Could be a list, could be a string!
    if not isinstance(param_types, list):
        param_types = [param_types]
    
    def decorator(func):
        """The actual decorator"""

        @wraps(func)
        def wrapper(self, params, idx):
            """The function wrapper"""
            # Setup-------------------------------------------------------------
            # Check if the cache_params dictionary has been initialized
            if not hasattr(self, '_cache_params'):
                raise ValueError("'_cache_params' has not been initialized. "+\
                                 "Use the 'set_cache_params' function to initialize it.")
            
            # Make _cache_{func.__name__} if it doesn't exist
            cache_func_name = f'_cache_{func.__name__}'
            if not hasattr(self, cache_func_name):
                toset = [[None,None] for _ in range(self._npsr)] # one key and value for each pulsar
                setattr(self, cache_func_name, toset)
                # Also setup cache stats
                setattr(self, f'{cache_func_name}_hit', np.zeros(self._npsr, dtype=int))
                setattr(self, f'{cache_func_name}_miss', np.zeros(self._npsr, dtype=int))

            # Actual caching----------------------------------------------------
            # Get the cache for this function and pulsar
            func_cache = getattr(self, cache_func_name)[idx]
            func_hit = getattr(self, f'{cache_func_name}_hit')[idx]
            func_miss = getattr(self, f'{cache_func_name}_miss')[idx]

            # Get all of the parameters of the given types
            cache_params = []
            for k in param_types:
                # Get the parameter names for this type and pulsar
                cache_params.extend(self._cache_params[k][idx])
            
            # Get a dictionary of the parameters we care about
            cache_values = []
            for k in cache_params:
                v = params[k]
                if np.ndim(v) == 0:
                    cache_values.append(v)
                else:
                    # Some parameters can be arrays, so convert to tuple for hashing
                    cache_values.append(tuple(v))


            # Construct a key from these parameters (must be tuples for hashing)
            cache_key = tuple(cache_values)

            # Find in cache-----------------------------------------------------
            if func_cache[0] == cache_key:
                # Cache hit!
                func_hit += 1
                return func_cache[1]
            else:
                # Cache miss!
                func_miss += 1
                # Call the function
                result = func(self, params, idx)
                # Update the cache
                func_cache[0] = cache_key
                func_cache[1] = result
                return result
        
        return wrapper
        # End of wrapper
    
    return decorator
    # End of decorator

            
def set_cache_params(obj, pta):
    """A function to set the parameters necessary for product_cache decorator

    This function initializes an attribute `_cache_params` which is a dictionary 
    of caching parameter sets and `_npsr` which is the number of pulsars in the PTA.

    The `_cache_params` dictionary contains three keys:
    - 'white_noise': a list of sets of white noise parameters for each pulsar
    - 'basis': a list of sets of basis parameters for each pulsar
    - 'delay': a list of sets of delay parameters for each pulsar

    This function assumes that the `_signalcollections` attribute of the PTA object
    contains additional attributes `white_params`, `basis_params`, and `delay_params`
    which are lists of parameter names for each pulsar.

    These parameter names must be the same as those used in the MCMC sampling!    

    Args:
        obj (Object): The object to set the cache parameters on.
        pta (enterprise.signals.signal_base.PTA): The PTA object.
    """
    obj._npsr = len(pta._signalcollections)
    wn, ba, de = [], [], []
    for psr in pta._signalcollections:
        wn.append(sorted(psr.white_params))
        ba.append(sorted(psr.basis_params))
        de.append(sorted(psr.delay_params))

    obj._cache_params = { 'white_noise': wn, 'basis': ba, 'delay': de }

