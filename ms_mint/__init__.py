"""A Python library for targetd metabolomics."""

import os
import logging
from ._version import get_versions
from .Mint import Mint

__version__ = get_versions()["version"]
del get_versions

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))

logging.info(Mint.version)
