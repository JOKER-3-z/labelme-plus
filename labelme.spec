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
    a.binaries,
    a.zipfiles,
    a.datas,
    name='labelme',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon='labelme/icons/icon.ico'
)

#coll = COLLECT(exe,
 #              a.binaries,
  #             a.zipfiles,
   #            a.datas,
    #           strip=False,
     #          upx=True,
      #         upx_exclude=[],
       #        name='labelPose')

app = BUNDLE(
    exe,
    name='Labelme.app',
    icon='labelme/icons/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
