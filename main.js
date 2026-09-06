const { app, BrowserWindow, ipcMain, screen } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const path = require("path");

let backend = null;

function overlayUrl(port) {
  const arg = process.argv.find((a) => /^https?:\/\//.test(a));
  if (arg) return arg.replace(/\/?$/, "/");
  return `http://127.0.0.1:${port || process.env.OVERLAY_PORT || "8765"}/`;
}

function probe(url, ms = 400) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve("ok");
    });
    req.setTimeout(ms, () => {
      req.destroy();
      resolve("stale");
    });
    req.on("error", (err) => {
      resolve(err.code === "ECONNREFUSED" ? "free" : "stale");
    });
  });
}

async function waitFor(url, tries = 120) {
  for (let i = 0; i < tries; i++) {
    if ((await probe(url, 500)) === "ok") return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("backend did not start: " + url);
}

async function pickPort(start) {
  const base = Number(start || process.env.OVERLAY_PORT || 8765);
  for (let port = base; port < base + 16; port++) {
    const url = `http://127.0.0.1:${port}/`;
    const state = await probe(url);
    if (state === "free") return { url, port };
    console.log("skipping busy overlay port", port);
  }
  throw new Error("no free overlay port");
}

function spawnBackend(url) {
  const port = new URL(url).port || "8765";
  const env = {
    ...process.env,
    COPILOT_EMBEDDED: "1",
    OVERLAY: "0",
    OVERLAY_PORT: String(port),
  };
  const name = process.platform === "win32" ? "copilot-backend.exe" : "copilot-backend";
  const packed = path.join(process.resourcesPath, name);
  if (app.isPackaged && fs.existsSync(packed)) {
    const data = app.getPath("userData");
    env.COPILOT_DATA_DIR = data;
    env.TRANSCRIBE_LOG_DIR = path.join(data, "log");
    backend = spawn(packed, ["--no-overlay"], { env, cwd: data });
  } else {
    const root = __dirname;
    const py = path.join(root, "venv", "bin", process.platform === "win32" ? "python.exe" : "python");
    const exe = fs.existsSync(py) ? py : "python3";
    backend = spawn(exe, [path.join(root, "main.py"), "--no-overlay"], { env, cwd: root });
  }
  backend.stdout?.on("data", (buf) => process.stdout.write(buf));
  backend.stderr?.on("data", (buf) => process.stderr.write(buf));
  backend.on("exit", (code) => {
    if (code && code !== 0) console.error("backend exited", code);
  });
}

function createWindow(url) {
  const area = screen.getPrimaryDisplay().workAreaSize;
  const width = 420;
  const height = 640;
  const win = new BrowserWindow({
    width,
    height,
    x: Math.max(0, area.width - width - 18),
    y: 18,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    hasShadow: true,
    backgroundColor: "#00000000",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });
  win.setContentProtection(true);
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  win.webContents.on("did-fail-load", (_e, code, desc) => {
    console.error("load failed", code, desc);
  });
  win.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown") return;
    const quitKey = input.key === "w" || input.key === "q" || input.key === "W" || input.key === "Q";
    if (quitKey && (input.meta || input.control)) {
      event.preventDefault();
      app.quit();
    }
  });
  win.loadURL(url);
  const sendFs = () => {
    if (!win.isDestroyed()) win.webContents.send("copilot-fullscreen", win.isFullScreen());
  };
  win.on("enter-full-screen", sendFs);
  win.on("leave-full-screen", sendFs);
}

app.whenReady().then(async () => {
  const handedOff = process.argv.some((a) => /^https?:\/\//.test(a));
  let url = overlayUrl();
  console.log("overlay", url);
  try {
    if (handedOff) {
      await waitFor(url);
    } else {
      const picked = await pickPort();
      url = picked.url;
      if (picked.port !== Number(process.env.OVERLAY_PORT || 8765)) {
        console.log("port busy, using", picked.port);
      }
      console.log("starting python backend…");
      spawnBackend(url);
      await waitFor(url);
      console.log("backend ready");
    }
  } catch (err) {
    console.error(err.message);
  }
  createWindow(url);
  console.log("window open — look at the top-right of the screen (no dock icon)");
});

let sidebarBounds = null;

ipcMain.on("copilot-sidebar", (e, on) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (!win || win.isDestroyed() || win.isFullScreen()) return;
  const extra = 280;
  if (on) {
    const cur = win.getBounds();
    if (!sidebarBounds) sidebarBounds = { ...cur };
    const area = screen.getPrimaryDisplay().workArea;
    const width = Math.min(sidebarBounds.width + extra, area.width);
    const x = Math.max(area.x, sidebarBounds.x + sidebarBounds.width - width);
    win.setBounds({ x, y: sidebarBounds.y, width, height: sidebarBounds.height });
    return;
  }
  if (sidebarBounds) {
    win.setBounds(sidebarBounds);
    sidebarBounds = null;
  }
});

ipcMain.on("copilot-quit", () => app.quit());
ipcMain.on("copilot-minimize", (e) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  win?.minimize();
});
ipcMain.on("copilot-fullscreen", (e) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (!win) return;
  win.setFullScreen(!win.isFullScreen());
});
ipcMain.handle("copilot-fullscreen-state", (e) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  return !!win?.isFullScreen();
});
app.on("before-quit", () => {
  if (backend && !backend.killed) backend.kill();
});
app.on("window-all-closed", () => app.quit());
