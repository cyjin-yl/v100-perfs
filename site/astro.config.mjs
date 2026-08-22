import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";

// GitHub Pages 子站: https://cyjin-yl.github.io/v100-perfs/
export default defineConfig({
  site: "https://cyjin-yl.github.io",
  base: "/v100-perfs",
  output: "static",
  integrations: [react(), sitemap()],
  markdown: { shikiConfig: { theme: "one-dark-pro" } },
});
