# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows
block_cipher = None

a = Analysis(
    ['labelme/__main__.py',"EISeg/eiseg/__main__.py"],
    pathex=['labelme','EISeg/eiseg','EISeg'], #,
    binaries=[], #'/usr/local/lib/python3.6/dist-packages/paddle/libs/'
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons/*', 'labelme/icons'),
        ('labelme/weights/*', "labelme/weights/"),
                ('EISeg/eiseg/config/*', 'eiseg/config'),
                 ('C:/Users/dongx/anaconda3/envs/labelme/Lib/site-packages/paddle/libs/*', '.'),
    ],
    hiddenimports=[],#'paddle'
    hookspath=[],#'./hooks/'
    runtime_hooks=[],
    excludes=[]
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='win',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
          embed_manifest=False,
    codesign_identity=None,
    entitlements_file=None,
        icon='labelme/icons/icon.ico',
    uac_admin=False,
)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='labelPose')

#app = BUNDLE(
#    exe,
#    name='Labelme.app',
#    icon='labelme/icons/icon.icns',
#    bundle_identifier=None,
#    info_plist={'NSHighResolutionCapable': 'True'},
#)
