
from defiant.core import OptimalStatistic
from defiant.utils import check_jax_64bit, enable_jax_64bit

if not check_jax_64bit(warn=False):
    msg = "Defiant needs JAX 64-bit mode for numerical stability. Attempting to enable..."
    print(msg)
    try:
        enable_jax_64bit()
        print('Success!')
    except RuntimeError as e:
        print(f"Error occurred while enabling JAX 64-bit mode: {e}")


defiant_ascii = r"""
______ ___________ _____  ___   _   _ _____ 
|  _  \  ___|  ___|_   _|/ _ \ | \ | |_   _|
| | | | |__ | |_    | | / /_\ \|  \| | | |  
| | | |  __||  _|   | | |  _  || . ` | | |  
| |/ /| |___| |    _| |_| | | || |\  | | |  
|___/ \____/\_|    \___/\_| |_/\_| \_/ \_/                                          
"""

defiant_ascii_font_cred = """
Font 'DOOM' by Frans P. de Vries <fpv@xymph.iaf.nl>  18 Jun 1996
based on Big by Glenn Chappell 4/93 -- based on Standard
figlet release 2.1 -- 12 Aug 1994
"""