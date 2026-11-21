"""Compatibility exports for the shared protected runtime-file utilities.

Existing managed service imports retain the same functions and lock class.
Provisioning can use the common module without importing the plugin subsystem.
"""

from skulk.utils.runtime_files import (
    RuntimeLock as RuntimeLock,
)
from skulk.utils.runtime_files import (
    acquire_runtime_lock as acquire_runtime_lock,
)
from skulk.utils.runtime_files import (
    grp as grp,
)
from skulk.utils.runtime_files import (
    is_desktop_metadata as is_desktop_metadata,
)
from skulk.utils.runtime_files import (
    private_directory as private_directory,
)
from skulk.utils.runtime_files import (
    private_group,
)
from skulk.utils.runtime_files import (
    pwd as pwd,
)
from skulk.utils.runtime_files import (
    read_private as read_private,
)
from skulk.utils.runtime_files import (
    remove_private as remove_private,
)
from skulk.utils.runtime_files import (
    writable_only_by as writable_only_by,
)
from skulk.utils.runtime_files import (
    write_private as write_private,
)

_private_group = private_group
