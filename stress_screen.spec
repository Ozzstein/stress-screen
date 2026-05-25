# stress_screen.spec
# Build:
#   macOS:   pyinstaller stress_screen.spec  →  dist/stress_screen
#   Windows: pyinstaller stress_screen.spec  →  dist/stress_screen.exe
#
# Kaleido bundling: After `pip install kaleido`, find kaleido's executable:
#   python -c "import kaleido; print(kaleido.__file__)"
# Add the kaleido executable dir to datas and binaries as documented at
# https://github.com/plotly/Kaleido — then rebuild.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys, os

block_cipher = None

a = Analysis(
    ['src/stress_screen/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('configs/temp_mapping.yaml', 'configs'),
        ('src/stress_screen/reports/templates', 'stress_screen/reports/templates'),
        # Add kaleido binaries here when needed (see comment above)
    ],
    hiddenimports=collect_submodules('stress_screen'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stress_screen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)
