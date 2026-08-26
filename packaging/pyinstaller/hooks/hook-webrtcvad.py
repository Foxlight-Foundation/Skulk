"""PyInstaller hook for the ``webrtcvad-wheels`` distribution.

The upstream contrib hook assumes that the import package and distribution are
both named ``webrtcvad``. Skulk intentionally depends on the maintained
``webrtcvad-wheels`` distribution, which exports the same import package under
a different metadata name.
"""

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")
