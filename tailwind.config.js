/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./static-pages/*.html",
  ],
  theme: {
    fontFamily: {
      'mono': ['IBM Plex Sans', 'ui-monospace', 'SFMono-Regular'],
    },
    extend: {
      colors: {
        'vimh': '#8CB9F0'
      }
    },
  },
  plugins: [],
}

