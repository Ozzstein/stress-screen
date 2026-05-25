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

datas = [('configs/temp_mapping.yaml', 'configs')]
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
