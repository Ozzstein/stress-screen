# stress_screen.spec
# Build:
#   macOS:   pyinstaller stress_screen.spec  →  dist/stress_screen
#   Windows: pyinstaller stress_screen.spec  →  dist/stress_screen.exe
#
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys, os

datas = [
    ('configs/temp_mapping.yaml', 'configs'),
    ('configs/analysis_defaults.yaml', 'configs'),
]

# Kaleido static-image engine (kaleido 0.2.x): bundle its Chromium-based
# executable directory, otherwise PDF export dies in the frozen binary.
import kaleido
_kaleido_exe_dir = os.path.join(os.path.dirname(kaleido.__file__), 'executable')
if not os.path.isdir(_kaleido_exe_dir):
    raise SystemExit(
        f"kaleido executable dir not found at {_kaleido_exe_dir}; "
        "PDF export would be broken in the frozen binary"
    )
datas.append((_kaleido_exe_dir, 'kaleido/executable'))
templates_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src', 'stress_screen', 'reports', 'templates')
if os.path.isdir(templates_dir):
    datas.append((templates_dir, 'reports/templates'))

a = Analysis(
    ['src/stress_screen/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules('stress_screen'),
    hookspath=[],
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
    [],
    name='stress_screen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)
