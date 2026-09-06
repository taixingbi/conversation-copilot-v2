# -*- mode: python ; coding: utf-8 -*-
# Output: dist-backend/copilot-backend  (pass --distpath dist-backend)

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "ui/overlay.html"), "ui"),
    (str(ROOT / "prompt/qa_instructions.json"), "prompt"),
    (str(ROOT / "profile/example.md"), "profile"),
]
binaries = []
hidden = [
    "events",
    "memory",
    "metrics",
    "noise",
    "questions",
    "speakers",
    "stt",
    "llm",
    "llm.answer",
    "llm.client",
    "llm.prompt",
    "ui",
    "ui.server",
]
for pkg in ("sounddevice", "sherpa_onnx", "numpy", "pywhispercpp"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="copilot-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
