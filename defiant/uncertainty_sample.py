

from defiant.utils import get_PRNGKey
from defiant.jitting import jit, jit_method

import jax.numpy as jnp
import jax.random as jrandom


def uncertainty_sample(A2, Var_A2, pfos=False, mcos=False, n_usamples=100):
    key = get_PRNGKey()
    if not pfos and not mcos:
        return _nm_os_unc_samp_(A2, Var_A2, n_usamples, key)
    if pfos and not mcos:
        return _nm_pfos_unc_samp_(A2, Var_A2, n_usamples, key)
    if not pfos and mcos:
        return _nm_mcos_unc_samp_(A2, Var_A2, n_usamples, key)
    if pfos and mcos:
        return _nm_pfmcos_unc_samp_(A2, Var_A2, n_usamples, key)
    raise ValueError(f"Invalid combination of pfos and mcos flags: pfos={pfos}, mcos={mcos}")


@jit(static_argnums=2)
def _nm_os_unc_samp_(A2, Var_A2, usamples, key):
    nm_iter = A2.shape[0]

    # Create uniform normal random numbers [nm_iter, usamples]
    U = jrandom.normal(key, shape=(nm_iter, usamples))
    # Scale and shift these into the proper space
    all_A2 = A2[:,None] + U * jnp.sqrt(Var_A2)[:,None] # [nm_iter, usamples]
    # Reshape into single array of samples
    all_A2 = all_A2.reshape(-1) # [nm_iter*n_usamples]
    return all_A2

@jit(static_argnums=2)
def _nm_pfos_unc_samp_(Sk, Var_Sk, usamples, key):
    nm_iter, n_freq = Sk.shape

    # Create uniform normal random numbers [nm_iter, n_freq, usamples]
    U = jrandom.normal(key, shape=(nm_iter, n_freq, usamples))
    # Scale and shift these into the proper space
    all_Sk = Sk[:,:,None] + U * jnp.sqrt(Var_Sk)[:,:,None] # [nm_iter, n_freq, usamples]
    # Collapse the nm_iter and usamples axes
    all_Sk = all_Sk.swapaxes(1,2).reshape(-1, n_freq) # [nm_iter*n_usamples, n_freq]
    return all_Sk
    
@jit(static_argnums=2)
def _nm_mcos_unc_samp_(A2, Var_A2, usamples, key):
    nm_iter, n_orf = A2.shape

    # Create uniform normal random numbers, [nm_iter, n_orf, usamples]
    U = jrandom.normal(key, shape=(nm_iter, n_orf, usamples)) 
    # Take cholesky decomposition of covariance matrices (first axes is batch dimension)
    L = jnp.linalg.cholesky(Var_A2) # [nm_iter, n_orf, n_orf]
    # Scale and shift these into the proper space
    all_A2 = A2[:,:,None] + L @ U # [nm_iter, n_orf, usamples]
    # Collapse the nm_iter and usamples axes
    all_A2 = all_A2.swapaxes(1,2).reshape(-1, n_orf) # [nm_iter*n_usamples, n_orf]
    return all_A2

@jit(static_argnums=2)
def _nm_pfmcos_unc_samp_(Sk, Var_Sk, usamples, key):
    nm_iter, n_freq, n_orf = Sk.shape

    # Create uniform normal random numbers [nm_iter, n_freq, n_orf, usamples]
    U = jrandom.normal(key, shape=(nm_iter, n_freq, n_orf, usamples))
    # Take cholesky decomposition of covariance matrices (first 2 axes are batch dimensions)
    L = jnp.linalg.cholesky(Var_Sk) # [nm_iter, n_freq, n_orf, n_orf]
    # Scale and shift these into the proper space
    all_Sk = Sk[:,:,:,None] + L @ U # [nm_iter, n_freq, n_orf, n_usamples]
    # Reshape into single array
    all_Sk = all_Sk.swapaxes(1,3).reshape(-1, n_orf, n_freq) # [nm_iter*n_usamples, n_orf, n_freq]
    # Reshape again into [nm_iter*n_usamples, n_freq, n_orf]
    all_Sk = jnp.swapaxes(all_Sk,1,2)
    return all_Sk


