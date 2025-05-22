
class BadCoreError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    la_forge.core.Core() usage in the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'InvalidCoreError: {self.message}'
    
class BadParametersError(Exception):
    """A Simple Exception class for more specific error messages relating to bad
    parameter usage in the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'BadParameterError: {self.message}'
    
class ModelPTAError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    enterprise PTA object usage in the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'ModelPTAError: {self.message}'
    
class NMOSInteruptError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    noise marginalization of the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'NMOSIteruptionError: {self.message}'
    
class PCOSInteruptError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    pair covaraiance of the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'InteruptedPairCovariance: {self.message}'
    
class ORFNotFoundError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    ORF functions in the DEFIANT
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'ORFNotFoundError: {self.message}'
    
class NaNPairwiseError(Exception):
    """A Simple Exception class for more specific error messages relating to the 
    NaN values in the pair-wise uncertainties

    This exception occurs when there are NaN values found in the pairwise uncertainties,
    either sig_ab or sig_ab(f_k). This most often occurs when the individual pulsar Z 
    matrices (computed in _compute_XZ() method) have NaN values. 

    The most common problem which causes this error is when the pulsar noise models are 
    invalid for a particular set of parameter values. For instance, if your pulsar 
    noise parameters are outside of the prior bounds in the PTA noise model.
    
    i.e. check your intrinsic red noise amplitude, spectral index. If these are outside 
    of the prior of the PTA model, then Z will be non-positive definite and will result
    in NaN values in the pair-wise uncertainties 
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return f'NaNPairwiseError: {self.message}'

    def extended_response(): 
        import textwrap

        text  = 'DEFIANT is about to throw an exception:\n\n'
        text += 'This exception occurs when there are NaN values found in the pairwise uncertainties, ' 
        text += 'either sig_ab or sig_ab(f_k). This most often occurs when the individual pulsar Z '
        text += 'matrices (computed in _compute_XZ() method) have NaN values. \n\n' 

        text += 'The most common problem which causes this error is when the pulsar noise models are '
        text += 'invalid for a particular set of parameter values. For instance, if your pulsar '
        text += 'noise parameters are outside of the prior bounds in the PTA noise model. \n\n'
    
        text += 'TLDR: check your intrinsic red noise amplitude, spectral index. If these are outside '
        text += 'of the prior of the PTA model, then Z will be non-positive definite and will result '
        text += 'in NaN values in the pair-wise uncertainties. If you are using particularly noisy pulsars, '
        text += 'you may want to try using clip_z in the constructor.'
        return textwrap.fill(text,width=80)