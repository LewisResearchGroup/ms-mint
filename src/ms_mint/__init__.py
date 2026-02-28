import logging
import os

from ._version import __version__
from .Mint import Mint

__all__ = ["__version__"]

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))

Mint.version = __version__

logging.info(Mint.version)
