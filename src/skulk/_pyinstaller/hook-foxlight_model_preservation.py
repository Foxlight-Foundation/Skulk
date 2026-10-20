"""Include resources that the preservation verifier reads through importlib."""

from skulk._pyinstaller import preservation_data_files

datas = preservation_data_files()
