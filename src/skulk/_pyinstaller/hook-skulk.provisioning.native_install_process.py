"""Include the stdlib guardian source when freezing native installer support."""

from skulk._pyinstaller import native_installer_data_files

datas = native_installer_data_files()
