import logging
import os
import sys
import webbrowser

from skulk.shared.constants import SKULK_CONFIG_HOME

logger = logging.getLogger(__name__)

_FIRST_RUN_MARKER = SKULK_CONFIG_HOME / ".dashboard_opened"
_SSH_ENVIRONMENT_VARIABLES = ("SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY")


def _is_first_run() -> bool:
    return not _FIRST_RUN_MARKER.exists()


def _mark_first_run_done() -> None:
    _FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _FIRST_RUN_MARKER.touch()


def _should_open_dashboard() -> bool:
    """Return whether this process should open the dashboard in a GUI browser.

    Browser auto-open is reserved for a local interactive terminal. Remote SSH
    commands and non-interactive service or qualification launches can still
    have access to a logged-in macOS GUI session, where ``webbrowser.open``
    would otherwise accumulate tabs in the user's desktop browser.
    """

    return (
        not os.environ.get("SKULK_RUNTIME_DIR")
        and not any(os.environ.get(name) for name in _SSH_ENVIRONMENT_VARIABLES)
        and sys.stderr.isatty()
    )


def print_startup_banner(port: int) -> None:
    """Print the startup banner and open the dashboard on local first runs.

    Args:
        port: Local HTTP port serving the Skulk dashboard.

    Side effects:
        Writes the banner to stderr, may open the dashboard in the default
        browser for an interactive local first run, and records that first-run
        handling has completed.
    """

    dashboard_url = f"http://localhost:{port}"
    first_run = _is_first_run()
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗  ██╗██╗   ██╗██╗     ██╗  ██╗                           ║
║   ██╔════╝██║ ██╔╝██║   ██║██║     ██║ ██╔╝                           ║
║   ███████╗█████╔╝ ██║   ██║██║     █████╔╝                            ║
║   ╚════██║██╔═██╗ ██║   ██║██║     ██╔═██╗                            ║
║   ███████║██║  ██╗╚██████╔╝███████╗██║  ██╗                           ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝                          ║
║                                                                       ║
║   Distributed AI Inference Cluster                                    ║
║   by Foxlight Foundation                                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  🌐 Skulk Dashboard & API Ready                                       ║
║                                                                       ║
║  {dashboard_url}{" " * (69 - len(dashboard_url))}║
║                                                                       ║
║  Click the URL above to open the Skulk dashboard                      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

"""

    print(banner, file=sys.stderr)

    if first_run and _should_open_dashboard():
        try:
            webbrowser.open(dashboard_url)
            logger.info("First run detected — opening dashboard in browser")
        except Exception:
            logger.debug("Could not auto-open browser", exc_info=True)
    if first_run:
        _mark_first_run_done()
