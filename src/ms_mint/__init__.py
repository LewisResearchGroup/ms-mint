import os
import logging
from .Mint import Mint
from ._version import __version__

__all__ = ["__version__"]

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))

Mint.version = __version__

logging.info(Mint.version)

