const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("copilot", {
  quit: () => ipcRenderer.send("copilot-quit"),
  minimize: () => ipcRenderer.send("copilot-minimize"),
  toggleFullscreen: () => ipcRenderer.send("copilot-fullscreen"),
  isFullscreen: () => ipcRenderer.invoke("copilot-fullscreen-state"),
  onFullscreen: (cb) => {
    ipcRenderer.on("copilot-fullscreen", (_e, on) => cb(!!on));
  },
  sidebar: (on) => ipcRenderer.send("copilot-sidebar", !!on),
});
