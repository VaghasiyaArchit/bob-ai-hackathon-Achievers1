/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#F7F1DE', // Light beige/cream
        surface: '#FFFFFF',    // Pure White
        border: '#E8E5DF',     // Warm subtle border
        text: '#4E220F',       // Dark espresso brown
        textMuted: '#9D6638',  // Warm brown (using as muted text or accents)
        primary: '#4E220F',    // Deep espresso
        danger: '#9D6638',     // Terracotta/Warm brown
        warning: '#B0BA99',    // Muted olive green
        caution: '#B0BA99',    // Muted olive green
        success: '#B0BA99',    // Muted olive green
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
