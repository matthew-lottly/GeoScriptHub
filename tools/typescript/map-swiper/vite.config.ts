import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "MapSwiper",
      fileName: "map-swiper",
      formats: ["es", "umd"],
    },
    rollupOptions: {
      external: ["maplibre-gl"],
      output: {
        globals: {
          "maplibre-gl": "maplibregl",
        },
      },
    },
  },
  plugins: [dts({ rollupTypes: true })],
  test: {
    environment: "happy-dom",
    globals: true,
  },
});
