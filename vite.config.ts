import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      // Copies src/assets and static script.js into dist/ so direct
      // browser opening and production builds both resolve cleanly.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          const srcAssets = path.resolve(__dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
          const scriptSrc = path.resolve(__dirname, 'script.js');
          const scriptDest = path.resolve(distDir, 'script.js');
          if (fs.existsSync(distDir) && fs.existsSync(scriptSrc)) {
            fs.copyFileSync(scriptSrc, scriptDest);
          }
          const modelSrc = path.resolve(__dirname, 'model-data.js');
          const modelDest = path.resolve(distDir, 'model-data.js');
          if (fs.existsSync(distDir) && fs.existsSync(modelSrc)) {
            fs.copyFileSync(modelSrc, modelDest);
          }
        },
      },
    ],
    build: {
      rollupOptions: {
        input: {
          main: path.resolve(__dirname, 'index.html'),
          prediction: path.resolve(__dirname, 'prediction.html'),
        },
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
