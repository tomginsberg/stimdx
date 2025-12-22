from ._core import Circuit
from ._cond import LastMeas, Cond, MeasParity
from ._execution import DynamicSampler

# Check if C++ extension is available
try:
    from ._stimdx_cpp import get_version as _cpp_version

    __cpp_available__ = True
    __cpp_version__ = _cpp_version()
except ImportError:
    __cpp_available__ = False
    __cpp_version__ = None

__all__ = ["Circuit", "LastMeas", "Cond", "MeasParity", "DynamicSampler"]
__version__ = "0.2.0"
