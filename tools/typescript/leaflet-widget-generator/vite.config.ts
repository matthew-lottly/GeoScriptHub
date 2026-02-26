import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "LeafletWidgetGenerator",
      // PLACEHOLDER: adjust the output filename prefix if you rename this package
      fileName: "leaflet-widget-generator",
      formats: ["es", "umd"],
    },
    rollupOptions: {
      // Leaflet is a peer dependency — exclude it from the bundle
      external: ["leaflet"],
      output: {
        globals: {
          leaflet: "L",
        },
      },
    },
  },
  plugins: [dts({ rollupTypes: true })],
  test: {
    // Vitest configuration
    environment: "jsdom",
    globals: true,
  },
});
