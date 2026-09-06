# -*- mode: python ; coding: utf-8 -*-
# Output: dist-backend/copilot-backend  (pass --distpath dist-backend)

from PyInstaller.utils.hooks import collect_all

datas = [
    ("ui/overlay.html", "ui"),
    ("prompt/qa_instructions.json", "prompt"),
    ("profile/example.md", "profile"),
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
    ["main.py"],
    pathex=[],
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
