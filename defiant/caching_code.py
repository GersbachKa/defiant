
import numpy as np

def product_cache(param_types, save_stats=False):
    """A decorator to cache the results of a matrix product based on the parameters and pulsar

    The cache is stored in the `_cache_{func.__name__}` attribute of the class instance.
    The cache only stores 1 result per pulsar per function!
    
    The cache itself is a list of lists, where the outer list corresponds to pulsars,
    and each inner list contains two elements: the cache key and the result of the function.
    The cache key is a tuple of key-value pairs, where the keys are the names of the parameters
    and the values are the corresponding parameter values for that pulsar.

    The decorator also keeps track of cache hits and misses if `save_stats` is set to True.
    The statistics are stored in the `_cache_stats` attribute of the class instance,
    which is a list of two integers: the number of hits and the number of misses.

    Valid parameter types to cache are:
    - 'white_noise': white noise parameters (efac, equad, etc.)
    - 'basis': basis parameters (For the fourier basis, chromatic noise, etc.)
    - 'delay': delay parameters (For deterministic signals, such as a continuous wave)

    Args:
        param_types (list): A list of parameter types to cache.
        save_stats (bool, optional): Whether to save cache statistics. Defaults to False.

    Raises:
        ValueError: If `param_types` is not a list or if `_cache_params` has not been initialized.
        ValueError: If `param_types` contains invalid parameter types.

    Returns:
        function: The decorated function.
    """

    if not isinstance(param_types, list):
        param_types = [param_types]
    
    def decorator(func):
        """A caching decorator"""

        def wrapper(self, idx, params):
            """The wrapper function that performs the caching"""

            # Setup-------------------------------------------------------------
            # Check if the cache_params dictionary has been initialized
            if not hasattr(self, '_cache_params'):
                raise ValueError("'_cache_params' has not been initialized. "+\
                                 "Use the 'set_cache_params' function to initialize it.")
            
            # Make _cache_{func.__name__} if it doesn't exist
            cache_func_name = f'_cache_{func.__name__}'
            if not hasattr(self, cache_func_name):
                setattr(self, cache_func_name, [[None,None] for _ in range(self.npsr)])

            # Make stats if save_stats is True
            if save_stats and not hasattr(self, f'_cache_stats'):
                setattr(self, f'_cache_stats', [0,0]) # (hits, misses)

            # Get the cache for this function, for this pulsar
            func_cache = getattr(self, cache_func_name)[idx]

            # Make the cache_key------------------------------------------------
            # Get all of the dictionary keys of the parameters of the given types
            keys = []
            for k in param_types:
                keys.extend(self._cache_params[k][idx])
            
            # Get the key value pairs for the parameters from params
            key_vals = []
            for k in keys:
                p = params[k]
                if np.ndim(p) == 0:
                    key_vals.append((k, p))
                else:
                    key_vals.append((k, tuple(p))) # Some parameters can be arrays

            # Construct the cache key
            cache_key = tuple(key_vals)

            # Find in cache-----------------------------------------------------
            if cache_key == func_cache[0]:
                # Cache hit!
                if save_stats:
                    self._cache_stats[0] += 1
                return func_cache[1]
            else:
                # Cache miss!
                if save_stats:
                    self._cache_stats[1] += 1

                # Call the function
                result = func(self, idx, params)

                # Update the cache
                func_cache[0] = cache_key
                func_cache[1] = result

                return result
        
        return wrapper
    
    return decorator

            
def set_cache_params(obj, pta):
    """A function to set the parameters necessary for product_cache decorator

    This function initializes two attributes into the object which you call it on:
    - `npsr`: the number of pulsars in the PTA
    - `_cache_params`: a dictionary of caching parameter sets

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
    npsr = len(pta._signalcollections)
    obj.npsr = npsr

    wn, ba, de = [], [], []
    for psr in pta._signalcollections:
        wn.append(set(psr.white_params))
        ba.append(set(psr.basis_params))
        de.append(set(psr.delay_params))


    obj._cache_params = { 'white_noise': wn, 'basis': ba, 'delay': de }

