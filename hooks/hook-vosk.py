from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# モデルやDLLを含める
datas = collect_data_files("vosk")
binaries = collect_dynamic_libs("vosk")
