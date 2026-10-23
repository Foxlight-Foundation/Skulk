"""Include resources that the preservation verifier reads through importlib."""

from importlib.util import find_spec

from skulk._pyinstaller import preservation_data_files

datas = preservation_data_files()

# The optional maintained adapter loads this native extension by module name.
# Its import has no peer-session side effect; unavailable platforms keep HTTP.
hiddenimports = ["libtorrent"] if find_spec("libtorrent") is not None else []

# CID parsing is lazy in the public optional adapter. Include its maintained
# implementation when the peer extra is present in a frozen Core build.
if find_spec("multiformats") is not None:
    hiddenimports.append("multiformats")
