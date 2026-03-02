"""Solara-based GUI for ms-mint.

This module provides an interactive GUI for ms-mint that works in Jupyter notebooks
and can be deployed as a standalone web application.

Requires the gui extra: pip install ms-mint[gui]

Example usage in Jupyter:
    ```python
    from ms_mint.gui import MintGui
    gui = MintGui()
    gui  # displays the app
    ```

Standalone deployment:
    ```bash
    solara run ms_mint.gui.app:MintGui
    ```
"""

try:
    from .app import MintGui
    from .state import MintState
    __all__ = ["MintGui", "MintState"]
except ImportError as e:
    raise ImportError(
        "GUI dependencies not installed. Install with: pip install ms-mint[gui]"
    ) from e
