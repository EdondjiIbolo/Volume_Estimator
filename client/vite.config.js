import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/predict": {
        target: "http://127.0.0.1:5001", // Dirección del backend Flask
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
