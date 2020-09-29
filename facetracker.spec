# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['facetracker.py'],
             pathex=['C:\\OpenSeeFaceBuild'],
             binaries=[('dshowcapture/dshowcapture_x86.dll', '.'), ('dshowcapture/dshowcapture_x64.dll', '.'), ('dshowcapture/libminibmcapture32.dll', '.'), ('dshowcapture/libminibmcapture64.dll', '.'), ('escapi/escapi_x86.dll', '.'), ('escapi/escapi_x64.dll', '.'), ('run.bat', '.'), ('msvcp140.dll', '.'), ('vcomp140.dll', '.'), ('concrt140.dll', '.'), ('vccorlib140.dll', '.')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'mpl-data', 'PyInstaller', 'pywt', 'skimage', 'scipy', 'pyinstaller'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

remove_bin = []
for bin in a.binaries:
    if bin[0].startswith("opencv_video") or bin[0].startswith("PyInstaller"):
        remove_bin.append(bin[0])
        print(bin)
remove_dat = []
for bin in a.datas:
    if bin[0].startswith("opencv_video") or bin[0].startswith("PyInstaller"):
        remove_dat.append(bin[0])
        print("data ", bin)

a.binaries = [x for x in a.binaries if not x[0] in remove_bin]
a.datas = [x for x in a.datas if not x[0] in remove_dat]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='facetracker',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='facetracker')
