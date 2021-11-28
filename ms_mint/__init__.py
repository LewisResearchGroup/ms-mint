import os
from ._version import get_versions
from .Mint import Mint

__version__ = get_versions()["version"]
del get_versions

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))
