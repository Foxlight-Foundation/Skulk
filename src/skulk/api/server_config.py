"""Production direct-API transport configuration shared with installed checks."""

from hypercorn.config import Config

from skulk.api.realtime import REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES
from skulk.shared.logging import InterceptLogger


def create_local_api_server_config(port: int) -> Config:
    """Return direct-API server settings without opening a listener.

    Args:
        port: The configured direct API port; zero permits an ephemeral bind.

    Returns:
        Hypercorn configuration admitting bounded native packets and realtime
        events. Individual endpoints retain their own smaller JSON limits.
    """
    configuration = Config()
    configuration.bind = [f"0.0.0.0:{port}"]
    # Keep the logging bridge aligned with shared.logging. This config is also
    # used by signed installed qualification so transport bounds cannot drift.
    configuration.accesslog = None
    configuration.errorlog = "-"
    configuration.logger_class = InterceptLogger
    configuration.websocket_max_message_size = REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES
    configuration.websocket_ping_interval = 20.0
    return configuration
