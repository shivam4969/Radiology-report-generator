import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// The API port follows PORT (same variable the server reads) so one .env drives both.
const API = `http://localhost:${process.env.PORT || 8787}`;

export default defineConfig({
  root: resolve(import.meta.dirname),
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API, timeout: 0, proxyTimeout: 0 },
      '/samples': { target: API, timeout: 0, proxyTimeout: 0 },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
