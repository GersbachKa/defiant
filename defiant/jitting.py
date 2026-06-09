
from jax import jit as jax_jit
from functools import wraps

def jit(func=None, static_argnums=None):
    """A wrapper/decorator to JIT compile functions while preserving metadata.

    This decorator can be used to JIT compile a function or method while
    preserving the original function's metadata using functools.wraps. It can be
    used in two ways:
    1. As a simple decorator: `@jit` or `@jit(static_argnums=...)`
    2. As a wrapper with arguments: `jit(func, static_argnums=...)`

    The `static_argnums` parameter allows you to specify which arguments should
    be treated as static (i.e., not traced by JAX) when JIT compiling the function.
    For class methods, you typically want to include `0` in `static_argnums` to treat
    `self` as static. 

    Parameters
    ----------
    func : callable, optional
        The function to be wrapped and JIT compiled, by default None
    static_argnums : tuple of ints, optional
        The indices of the arguments to be treated as static, by default None

    Returns
    -------
    callable
        The JIT compiled version of the input function with preserved metadata.
    """
    # Check if static_argnums is provided, if so, ensure it's a tuple
    if static_argnums is not None:
        if isinstance(static_argnums, int):
            # If it's a single integer, convert it to a tuple
            static_argnums = (static_argnums,)
        else:
            # Assume its already an iterable of integers, convert to tuple
            static_argnums = tuple(static_argnums)


    def decorator(f):
        # Jit the function with the specified static arguments
        jit_func = jax_jit(f, static_argnums=static_argnums)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return jit_func(*args, **kwargs)
    
        return wrapper
    
    if func is None:
        return decorator # Return the decorator if no function is provided
    else:
        return decorator(func) # Otherwise, apply the decorator to the function
    

def jit_method(func):
    """A wrapper to JIT compile class methods with static `self` while preserving metadata.

    This decorator is specifically designed for class methods. It JIT compiles the 
    method while treating `self` as a static argument. This is identical to
    using `@jit(static_argnums=0)` on a class method.

    Parameters
    ----------
    func : callable
        The class method to be wrapped and JIT compiled.

    Returns
    -------
    callable
        The JIT compiled version of the input class method with preserved metadata.
    """
    return jit(func, static_argnums=0)

