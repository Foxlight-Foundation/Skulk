"""Skulk — distributed MLX inference. Forked from exo; substantially diverged."""

import os as _os

# Loguru annotates exception tracebacks with the values of the variables on
# each displayed line unless a handler is added with ``diagnose=False``, and it
# reads the default for that option from this variable once, when it is first
# imported. A failing frame holds whatever it was working on: a request's text,
# a credential being stored. Skulk's node sinks pass the option explicitly
# (``skulk.shared.logging``); this covers every handler they do not add: the
# pre-installed one in command-line tools, in the plugin service and in a
# spawned runner's own interpreter, and any later ``logger.add`` that leaves
# the option out. It runs before any Skulk module can import loguru, and child
# processes inherit it. It is not an operator setting: an exported value is
# replaced, so no environment can turn a node's logs into a record of requests.
_os.environ["LOGURU_DIAGNOSE"] = "False"
