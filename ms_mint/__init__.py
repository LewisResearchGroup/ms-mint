import os
import logging
from .Mint import Mint

try:
    from .version import __version__
except ImportError:
    __version__ = "0.0.0"

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))

Mint.version = __version__

logging.info(Mint.version)
