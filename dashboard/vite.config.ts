import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const dashboardBase = process.env.VITE_DASHBOARD_BASE || '/dashboard/'

export default defineConfig({
  plugins: [react()],
  base: dashboardBase,
})
