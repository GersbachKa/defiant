
from .custom_exceptions import *
from .utils import matrix_product_trace as mpt
from defiant.jitting import jit

import jax.numpy as jnp
from jax import vmap


@jit
def calculate_pair_covariance(Z, phihat, phi, orf, norm_ab, case_idx):
    npsr, nmode = Z.shape[0:2]
    npair = npsr * (npsr - 1) // 2 # Number of unique pairs of pulsars (no self-pair)
    npop = npair * (npair + 1) // 2 # Number of unique pairs of pairs (with self-pair)

    # Pre-compute Z@diag(phi) for each pulsar for each phi
    Zphihat = phihat * Z # [npsr, nmode, nmode]
    Zphi = phi * Z # [npsr, nmode, nmode]

    # Pre-compute ZphihatZphi
    ZphihatZphi = jnp.einsum('aij,bjk->abik', Zphihat, Zphi) # [npsr, npsr, nmode, nmode]

    def case1(a,b,c,d): # (ab, cd)
        t1  = orf[a,c]*orf[b,d] * mpt(ZphihatZphi[b,a], ZphihatZphi[c,d])
        t2  = orf[a,d]*orf[b,c] * mpt(ZphihatZphi[b,a], ZphihatZphi[d,c])
        return (t1+t2)

    def case2(a,b,c): # (ab, ac)
        t1  = orf[b,c] * mpt(Zphihat[b], ZphihatZphi[a,c])
        t2  = orf[a,c]*orf[a,b] * mpt(ZphihatZphi[b,a], ZphihatZphi[c,a])
        return (t1+t2)
    
    def case3(a,b): # (ab, ab)
        t1  = mpt(Zphihat[b], Zphihat[a]) 
        t2  = orf[a,b]**2 * mpt(ZphihatZphi[b,a], ZphihatZphi[b,a])
        return (t1+t2)

    C1, C2, C3 = case_idx
    C_flat = jnp.zeros(npop)
    
    # Case 1
    out = vmap(case1)(C1[1],C1[2],C1[3],C1[4]) 
    C_flat = C_flat.at[C1[0]].set(out)  

    # Case 2
    out = vmap(case2)(C2[1],C2[2],C2[3])
    C_flat = C_flat.at[C2[0]].set(out)

    # Case 3
    out = vmap(case3)(C3[1],C3[2])
    C_flat = C_flat.at[C3[0]].set(out)

    # Include normalization
    i,j = jnp.triu_indices(npair, k=0)
    C_flat *= norm_ab[i] * norm_ab[j]

    # Unflatten into a matrix
    C = jnp.zeros((npair, npair))
    C = C.at[i,j].set(C_flat)
    C = C.at[j,i].set(C_flat) # Symmetrize
    return C


def calculate_case_idx(npsr):
    npair = npsr * (npsr - 1) // 2
    npop = npair * (npair + 1) // 2
    pair_idx = jnp.array(jnp.triu_indices(npsr, k=1), dtype=jnp.int32) # [2, npair]
    pop_idx = jnp.array(jnp.triu_indices(npair, k=0), dtype=jnp.int32) # [2, npop]
    
    i,j = pop_idx
    (a,b), (c,d) = pair_idx[:, i], pair_idx[:, j]

    # Case 1: (ab, cd) with all indices different
    m1 = jnp.argwhere((a!=c)&(a!=d) & (b!=c)&(b!=d))[:, 0]
    c1 = jnp.array([m1, a[m1], b[m1], c[m1], d[m1]], dtype=jnp.int32)

    # Case 2: (ab,ac) a is repeated, 4 permutations:
    # (ab, cd) with a==c 
    m2_1 = jnp.argwhere((a==c)&(b!=d))[:, 0]
    c2_1 = jnp.array([m2_1, a[m2_1], b[m2_1], d[m2_1]], dtype=jnp.int32)

    # (ab, cd) with a==d
    m2_2 = jnp.argwhere((a==d)&(b!=c))[:, 0]
    c2_2 = jnp.array([m2_2, a[m2_2], b[m2_2], c[m2_2]], dtype=jnp.int32)

    # (ab, cd) with b==c
    m2_3 = jnp.argwhere((a!=d)&(b==c))[:, 0]
    c2_3 = jnp.array([m2_3, b[m2_3], a[m2_3], d[m2_3]], dtype=jnp.int32)

    # (ab, cd) with b==d
    m2_4 = jnp.argwhere((a!=c)&(b==d))[:, 0]
    c2_4 = jnp.array([m2_4, b[m2_4], a[m2_4], c[m2_4]], dtype=jnp.int32)

    # Combine all case 2 permutations
    c2 = jnp.concatenate([c2_1, c2_2, c2_3, c2_4], axis=1, dtype=jnp.int32)

    # Case 3: (ab, cd) with a==c and b==d
    m3 = jnp.argwhere((a==c)&(b==d))[:, 0]
    c3 = jnp.array([m3, a[m3], b[m3]], dtype=jnp.int32)

    # Return them all
    out = (c1, c2, c3) # Each is a (npairs_in_case, n_indices) array
    return out # [5, npop]

