import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import WeatherClassification from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WeatherClassification />
  </StrictMode>,
)
