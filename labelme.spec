# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows


a = Analysis(
    ['labelme/__main__.py',"EISeg/eiseg/__main__.py"],
    pathex=['labelme','EISeg/eiseg','EISeg'], #,
    binaries=[], #'/usr/local/lib/python3.6/dist-packages/paddle/libs/'
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons/*', 'labelme/icons'),
        ('weights/*', "weights/"),
                ('EISeg/eiseg/config/*', 'eiseg/config'),
    ],
    hiddenimports=[],#'paddle'
    hookspath=[],#'./hooks/'
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='labelme',
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon='labelme/icons/icon.ico',
)
app = BUNDLE(
    exe,
    name='Labelme.app',
    icon='labelme/icons/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
