const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("copilot", {
  quit: () => ipcRenderer.send("copilot-quit"),
});
