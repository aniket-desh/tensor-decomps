"""
backend module.
provides tensor computation backends (numpy for sequential, ctf for distributed).
"""

from . import numpy_ext

# ctf is optional - only import if available
try:
    from . import ctf_ext
    _HAS_CTF = True
except ImportError:
    _HAS_CTF = False


def get_backend(name='numpy'):
    """
    get the specified tensor backend.
    
    args:
        name: 'numpy' or 'ctf'
        
    returns:
        backend module
    """
    if name == 'numpy':
        return numpy_ext
    elif name == 'ctf':
        if not _HAS_CTF:
            raise ImportError(
                "[error] ctf backend not available. install ctf first."
            )
        return ctf_ext
    else:
        raise ValueError(f"[error] unknown backend: {name}. use 'numpy' or 'ctf'")


__all__ = [
    'numpy_ext',
    'get_backend',
]

if _HAS_CTF:
    __all__.append('ctf_ext')

