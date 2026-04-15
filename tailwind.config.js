/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./static/*.html",
  ],
  theme: {
    fontFamily: {
      'mono': ['IBM Plex Sans', 'ui-monospace', 'SFMono-Regular'],
    },
    extend: {
      colors: {
        'vimh': '#FFD364'
      }
    },
  },
  plugins: [],
}
