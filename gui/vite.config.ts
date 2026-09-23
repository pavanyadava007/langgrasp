import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API and the simulation worker run on one port (8000 by default, GUI_PORT to override). In development
// Vite serves the app and proxies everything the worker owns, so the browser sees a single origin and no CORS
// rule has to be relaxed on a port-forwarded host.
const apiPort = process.env.GUI_PORT ?? "8000";
const target = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  plugins: [react()],
  // Built into the package so `python -m langgrasp.gui` serves it with no Node on the host.
  build: { outDir: "../langgrasp/gui/static", emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 900 },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target, changeOrigin: false },
      "/runs": { target, changeOrigin: false },
      "/ws": { target, ws: true, changeOrigin: false },
    },
  },
});
