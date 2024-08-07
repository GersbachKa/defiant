import numpy as np

from .custom_exceptions import ORFNotFoundError

################################################################################
#                       Creating your own ORF functions
#
# If you plan to make your own overlap reduction functions, you can use these
# functions as a template. Note that while the functions here are mostly vectorized,
# DEFIANT will not use vectorization for the ORF functions. The vectorization is
# done purely for the convenience of the user wanting to use the functions directly.
# Additionally, while the functions here allow for both inputs of enterprise.Pulsar
# objects and pulsar separations, DEFIANT will only call the orf functions with
# two enterprise.Pulsar objects. 
#
# This means that the minimum requirement for an ORF function is to take two
# enterprise.Pulsar objects as input and return a float.
# 
# Here is an example template which you can copy and modify:

def orf_template(psr1, psr2):
    """A simple template for creating your own ORF functions

    This is a simple template for creating your own ORF functions. The function
    will take two enterprise.Pulsar objects as input and return a float.

    Args:
        psr1 (enterprise.Pulsar): The first pulsar object
        psr2 (enterprise.Pulsar): The second pulsar object

    Returns:
        float: The ORF value
    """
    # It is also a good habit to include a check for the same pulsar. Technically,
    # DEFIANT should not supply two of the same pulsar, but this is still good practice.
    if psr1.name == psr2.name:
        return 1

    # If you have a custom isotropic ORF function, you can use the get_pulsar_separation
    # function to get the separation between the two pulsars. Anisotropic ORF functions
    # will may need specific pulsars properties to calculate the ORF value, hence the
    # need for the enterprise.Pulsar objects.
    xi = get_pulsar_separation(psr1, psr2)


    # Perform some calculations to get the ORF value. In this case, lets just return
    # the cosine of the separation. (A dipole ORF function)
    ret = np.cos(xi)

    return ret
################################################################################


# Useful ORF functions --------------------------------------------------------

def get_orf_function(orf_name='hd'):
    """A function to get the orf function from the defined_orfs list

    This function will return the orf function from the defined_orfs list. The 
    returned function will either need two enterprise.Pulsar objects OR a pulsar 
    separation in radians. If the orf_name is not found in the defined_orfs list, 
    then an ORFNotFoundError will be raised. All functions support vectorization.

    The list of pre-defined orfs are:
        - 'hd' or 'hellingsdowns': Hellings and Downs
        - 'dp' or 'dipole': Dipole
        - 'mp' or 'monopole': Monopole
        - 'gwdp' or 'gw_dipole': Gravitational wave dipole
        - 'gwmp' or 'gw_monopole': Gravitational wave monopole
        - 'st' or 'scalar_tensor': Scalar tensor

    Raises:
        ORFNotFoundError: If the orf_name is not found in the defined_orfs list

    Args:
        orf_name (str): The name of the pre-defined orf function to use. Defaults to 'hd'.
        separation_func (bool): _description_. Defaults to False.

    Returns:
        function: The orf function from the defined_orfs list
    """
    # Defied ORFs list is a bit weird. It is a list of tuples where the first element
    # is a list of names and the second element is the function. This is done so that
    # the function can be called by multiple names.

    for orf in defined_orfs: 
        if orf_name.lower() in orf[0]:
            return orf[1]
        
    # If the function is not found, raise an error
    raise ORFNotFoundError(f"The ORF function '{orf_name}' is not found in the defined_orfs list.")


def get_pulsar_separation(psr1, psr2):
    """A function to get the separation between two enterprise.Pulsar pulsars.

    This function will return the separation between two enterprise.Pulsar pulsars
    in radians. This function supports vectorization.

    Args:
        psr1 (enterprise.Pulsar or iterable): The first pulsar object(s)
        psr2 (enterprise.Pulsar or iterable): The second pulsar object(s)

    Returns:
        float or np.ndarray: The separation between the two pulsars in radians
    """
    if hasattr(psr1, '__iter__') and hasattr(psr2, '__iter__'):
        pos1 = np.array([p.pos for p in psr1])
        pos2 = np.array([p.pos for p in psr2])
        return np.arccos(np.dot(pos1, pos2))
    else:
        return np.arccos(np.dot(psr1.pos, psr2.pos))


# ORF functions ---------------------------------------------------------------


# Isotropic ORF functions -----------------------------------------------------

def orf_hd(arg1, arg2 = None):
    """The Hellings and Downs overlap reduction function

    This function will return the Hellings and Downs overlap reduction function using
    either two enterprise.Pulsar objects or a pulsar separation in radians. This
    function supports vectorization.

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    ret = np.zeros_like(xi)
    
    d = (1-np.cos(xi[xi!=0])) / 2
    ret[xi!=0] = (1/2) - (d/2) * ( (1/2) - 3*np.log(d) ) 
    
    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


def orf_dp(arg1, arg2 = None):
    """The Dipole overlap reduction function

    This function will return the dipole overlap reduction function using either 
    two enterprise.Pulsar objects or a pulsar separation in radians. This function 
    supports vectorization.

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    ret = np.cos(xi)

    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


def orf_mp(arg1, arg2 = None):
    """The Monopole overlap reduction function

    This function will return the monopole overlap reduction function using either 
    two enterprise.Pulsar objects or a pulsar separation in radians. This function
    supports vectorization.

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    if hasattr(xi, '__iter__'):
        ret = np.ones(len(xi))
    else:
        ret = 1

    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


def orf_gwdp(arg1, arg2 = None):
    """The Gravitational Wave Dipole overlap reduction function

    This function will return the gravitational wave dipole overlap reduction function 
    using either two enterprise.Pulsar objects or a pulsar separation in radians. 
    This function supports vectorization.

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    ret = np.cos(xi)/2

    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


def orf_gwmp(arg1, arg2 = None):
    """The Gravitational wave Monopole overlap reduction function

    This function will return the gravitational wave monopole overlap reduction 
    function using either two enterprise.Pulsar objects or a pulsar separation 
    in radians. This function supports vectorization.

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    ret = np.ones(len(xi))/2

    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


def orf_st(arg1, arg2 = None):
    """The scalar tensor overlap reduction function

    This function will return the scalar tensor overlap reduction function using 
    either two enterprise.Pulsar objects or a pulsar separation in radians. This
    function supports vectorization.

    Original author: Nima Laal from enterprise_extensions.model_orfs.py

    Args:
        arg1 (enterprise.Pulsar or float): The first pulsar object or the pulsar 
                separation in radians.
        arg2 (enterprise.Pulsar or None): The second pulsar object or None if arg1 
                is pulsar separation. Defaults to None.

    Returns:
        float: The ORF value
    """
    if arg2 is not None:
        # Two pulsar arguments
        xi = get_pulsar_separation(arg1, arg2)
    else:
        # One separation argument
        xi = arg1
    
    ret = 1/8 * (3.0 + np.cos(xi))

    # Check if the separation is zero (same pulsar)
    if hasattr(xi, '__iter__'):
        ret[xi == 0] = 1
    else:
        if xi == 0:
            ret = 1
    return ret


# Anisotropic ORF functions ---------------------------------------------------

# TODO: put in per-telescope orf functions

# Defined ORFs ----------------------------------------------------------------


defined_orfs = [
    (['hellingsdowns','hd'],    orf_hd),    # Hellings and Downs
    (['dipole','dp'],           orf_dp),    # Dipole
    (['monopole','mp'],         orf_mp),    # Monopole
    (['gw_dipole','gwdp'],      orf_gwdp),  # Gravitational wave dipole
    (['gw_monopole','gwmp'],    orf_gwmp),  # Gravitational wave monopole
    (['scalar_tensor','st'],    orf_st),    # Scalar tensor
]