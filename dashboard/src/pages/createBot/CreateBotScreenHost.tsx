import { useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { buildCreateBotPath } from './flowConfig'
import CreateBotActionDestinationPage from './CreateBotActionDestinationPage'
import CreateBotAdditionalSourcesPage from './CreateBotAdditionalSourcesPage'
import CreateBotDetailsPage from './CreateBotDetailsPage'
import CreateBotEmbedPage from './CreateBotEmbedPage'
import CreateBotProgressPage from './CreateBotProgressPage'
import CreateBotUrlsPage from './CreateBotUrlsPage'
import CreateBotWidgetPage from './CreateBotWidgetPage'

export default function CreateBotScreenHost() {
  const navigate = useNavigate()
  const location = useLocation()
  const { flow } = useCreateBotFlow()
  const { currentScreen, firstPath } = flow

  useEffect(() => {
    if (!currentScreen) {
      if (location.pathname !== firstPath) {
        navigate(firstPath, { replace: true })
      }
      return
    }
    const canonicalPath = buildCreateBotPath(currentScreen.path)
    if (location.pathname !== canonicalPath) {
      navigate(canonicalPath, { replace: true })
    }
  }, [currentScreen, firstPath, location.pathname, navigate])

  if (!currentScreen) return null

  switch (currentScreen.component) {
    case 'details':
      return <CreateBotDetailsPage />
    case 'source_urls':
      return <CreateBotUrlsPage />
    case 'additional_sources':
      return <CreateBotAdditionalSourcesPage />
    case 'training_progress':
      return <CreateBotProgressPage />
    case 'widget_design':
      return <CreateBotWidgetPage />
    case 'embed_install':
      return <CreateBotEmbedPage />
    case 'action_destination_url':
      return <CreateBotActionDestinationPage />
    default:
      return null
  }
}
