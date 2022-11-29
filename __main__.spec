# -*- mode: python ; coding: utf-8 -*-


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
                 ('C:/Users/dongx/anaconda3/envs/labelme/Lib/site-packages/paddle/libs/*', './'),
    ],
          #   pathex=[],
          #   binaries=[],
           #  datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='__main__',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='__main__')
