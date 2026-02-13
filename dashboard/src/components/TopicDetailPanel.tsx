import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import type { TopicUsageItem, TopicQuestionItem } from '../hooks/useDashboardData'

interface TopicDetailPanelProps {
  topic: TopicUsageItem | null
  botId: string
  onClose: () => void
}

function formatDateTime(iso: string): string {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

function capitalize(s: string) {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function TopicDetailPanel({ topic, botId, onClose }: TopicDetailPanelProps) {
  const { getTopicQuestions } = useDashboardData()
  const navigate = useNavigate()
  const [questions, setQuestions] = useState<TopicQuestionItem[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!topic) {
      setQuestions([])
      return
    }
    setLoading(true)
    getTopicQuestions(botId, topic.topic_id, 50)
      .then((res) => setQuestions(res?.questions ?? []))
      .finally(() => setLoading(false))
  }, [topic?.topic_id, botId])

  const handleQuestionClick = (q: TopicQuestionItem) => {
    navigate(`/bots/${botId}/conversations?session=${q.session_id}`)
    onClose()
  }

  const panelContent = (
    <AnimatePresence>
      {topic && (
        <>
          {/* Backdrop */}
          <motion.div
            className="panel-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          {/* Slide-out panel */}
          <motion.div
            className="topic-detail-panel"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          >
            {/* Header */}
            <div className="topic-panel-header">
              <div className="topic-panel-title-row">
                <div>
                  <div className="topic-panel-title">{capitalize(topic.topic)}</div>
                  <div className="topic-panel-meta">
                    <span className={`topic-category-badge cat-${topic.category}`}>
                      {capitalize(topic.category)}
                    </span>
                    {topic.origin && topic.origin !== 'extracted' && (
                      <span className="topic-origin-badge">{topic.origin === 'url_bank' ? 'URL bank' : topic.origin}</span>
                    )}
                    <span className="topic-panel-count">
                      {topic.question_count} question{topic.question_count !== 1 ? 's' : ''}
                    </span>
                  </div>
                </div>
                <button className="panel-close-btn" onClick={onClose} type="button" aria-label="Close">
                  ✕
                </button>
              </div>
              {topic.source_url && (
                <a
                  href={topic.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="topic-panel-source-link"
                >
                  <span>↗</span> View source page
                </a>
              )}
            </div>

            {/* Questions list */}
            <div className="topic-panel-body">
              <div className="topic-panel-section-title">Questions asked</div>
              {loading && (
                <div className="topic-panel-empty muted">Loading…</div>
              )}
              {!loading && questions.length === 0 && (
                <div className="topic-panel-empty muted">
                  No questions mapped to this topic yet. Try refreshing topics after more conversations.
                </div>
              )}
              {!loading && questions.length > 0 && (
                <div className="topic-questions-list">
                  {questions.map((q) => (
                    <button
                      key={q.id}
                      className="topic-question-row"
                      onClick={() => handleQuestionClick(q)}
                      type="button"
                    >
                      <div className="topic-question-text">
                        {q.question_text || '(no text)'}
                      </div>
                      <div className="topic-question-meta">
                        {q.session_title && (
                          <span className="topic-question-session">{q.session_title}</span>
                        )}
                        <span className="topic-question-time">{formatDateTime(q.asked_at)}</span>
                        <span className="topic-question-nav-icon">→</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )

  return createPortal(panelContent, document.body)
}
