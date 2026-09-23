import { defineConfig } from "vitest/config";

// Unit tests only. The Playwright spec under e2e/ matches vitest's default glob but needs a browser and a
// running worker, so it is run by `npm run e2e` instead.
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    exclude: ["node_modules", "e2e", "dist"],
    environment: "node",
    reporters: "default",
  },
});
