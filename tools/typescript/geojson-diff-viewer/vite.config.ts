import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [
    dts({ include: ["src"], insertTypesEntry: true }),
  ],
  build: {
    lib: {
      entry: "src/index.ts",
      name: "GeoJsonDiffViewer",
      fileName: "geojson-diff-viewer",
      formats: ["es", "umd"],
    },
    rollupOptions: {
      // Exclude Leaflet from the bundle — consumers must provide it.
      external: ["leaflet"],
      output: {
        globals: {
          leaflet: "L",
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/**/*.test.ts"],
  },
});
