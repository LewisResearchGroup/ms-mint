# -*- mode: python ; coding: utf-8 -*-
import os

src_dir = os.path.abspath(os.path.join(SPECPATH, os.pardir))
hooks_dir= os.path.join(src_dir, 'hooks')
script = os.path.join(src_dir, 'scripts', 'Mint.py')

a = Analysis(
    [script],
    pathex=[src_dir],
    hookspath=[hooks_dir],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='Mint',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Mint',
)
