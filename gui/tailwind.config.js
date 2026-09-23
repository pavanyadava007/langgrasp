/** Tailwind reads the design tokens from CSS variables (src/styles/tokens.css) so that one theme switch on
 *  <html data-theme> changes every utility class at once, and so the tokens stay readable in dev tools. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        edge: "var(--edge)",
        fg: "var(--fg)",
        "fg-muted": "var(--fg-muted)",
        accent: "var(--accent)",
        "accent-fg": "var(--accent-fg)",
        focus: "var(--focus)",
        ok: "var(--ok)",
        warn: "var(--warn)",
        fail: "var(--fail)",
        estop: "var(--estop)",
        "danger-bg": "var(--danger-bg)",
        "danger-fg": "var(--danger-fg)",
        running: "var(--running)",
        pending: "var(--pending)",
        gt: "var(--gt)",
      },
      borderRadius: { s: "var(--radius-s)", m: "var(--radius-m)", l: "var(--radius-l)" },
      spacing: { 1: "4px", 2: "8px", 3: "12px", 4: "16px", 5: "20px", 6: "24px", 8: "32px", 10: "40px", 12: "48px" },
      fontFamily: { sans: "var(--font-sans)", mono: "var(--font-mono)" },
      fontSize: {
        xs: ["12px", "16px"],
        sm: ["13px", "18px"],
        base: ["14px", "20px"],
        lg: ["16px", "24px"],
        xl: ["20px", "28px"],
        "2xl": ["28px", "36px"],
      },
    },
  },
  plugins: [],
};
