from ._version import get_versions
from .xray.absorption import Absori, Integrate_Absori

__version__ = get_versions()['version']
del get_versions
