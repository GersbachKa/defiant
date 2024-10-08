
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