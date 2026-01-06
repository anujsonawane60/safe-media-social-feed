import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0095f6',
        secondary: '#8e8e8e',
        background: '#fafafa',
        card: '#ffffff',
        border: '#dbdbdb',
        danger: '#ed4956',
      },
    },
  },
  plugins: [],
}
export default config
