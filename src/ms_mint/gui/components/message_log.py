"""Message log component for displaying status messages."""

from __future__ import annotations

from typing import Callable

import solara


@solara.component
def MessageLog(
    messages: solara.Reactive[list[str]],
    on_clear: Callable[[], None],
):
    """Component for displaying log messages.

    Args:
        messages: Reactive list of messages (newest first).
        on_clear: Callback to clear messages.
    """
    with solara.Card("Messages", margin=0):
        with solara.Column():
            # Clear button
            solara.Button(
                "Clear",
                on_click=on_clear,
                small=True,
                text=True,
                disabled=len(messages.value) == 0,
            )

            # Message area
            if messages.value:
                with solara.Column(
                    style={
                        "maxHeight": "200px",
                        "overflowY": "auto",
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                        "backgroundColor": "#f5f5f5",
                        "padding": "8px",
                        "borderRadius": "4px",
                    }
                ):
                    for msg in messages.value:
                        solara.Text(msg)
            else:
                solara.Text(
                    "No messages",
                    style={"color": "#999", "fontStyle": "italic"},
                )
