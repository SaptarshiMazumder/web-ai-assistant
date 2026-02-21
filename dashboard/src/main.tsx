import React from 'react'
import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import App from './App'
import './i18n'
import './style.css'

const env = (import.meta as { env: Record<string, string> }).env
const auth0Domain = env.VITE_AUTH0_DOMAIN || ''
const auth0ClientId = env.VITE_AUTH0_CLIENT_ID || ''
const auth0Audience = env.VITE_AUTH0_AUDIENCE || ''

const root = document.querySelector<HTMLDivElement>('#app')
if (!root) {
  throw new Error('Missing #app container')
}

createRoot(root).render(
  <React.StrictMode>
    <Auth0Provider
      domain={auth0Domain}
      clientId={auth0ClientId}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: auth0Audience,
      }}
      cacheLocation="localstorage"
      useRefreshTokens
    >
      <App />
    </Auth0Provider>
  </React.StrictMode>
)
