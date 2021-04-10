# -*- mode: python ; coding: utf-8 -*-

import os
import importlib

block_cipher = None


a = Analysis(['app.py'],
             pathex=['C:\\Users\\rainb\\env\\AI'],
             binaries=[],
             datas=[(os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__),
                              "lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so"),
                 "tensorflow/lite/experimental/microfrontend/python/ops/")],
             hiddenimports=['tensorflow.python.keras.engine.base_layer_v1','tensorflow.python.ops.while_v2','tensorflow.python.ops.numpy_ops'],
             hookspath=[],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
