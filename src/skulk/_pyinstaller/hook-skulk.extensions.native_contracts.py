"""Retain Core's lazy native reader, pinned cohort and public package identity."""

from skulk._pyinstaller import native_contract_data_files

datas = native_contract_data_files()
hiddenimports = ["_skulk_capability_kernel"]
