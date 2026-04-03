import { useState, useRef, useEffect } from 'react'

const API_URL = '/api'

const SUGGESTED_QUESTIONS = [
  "How do I register my address in Germany?",
  "What health insurance do I need as a student?",
  "How do I file a tax return (Steuererklärung)?",
  "What documents do I need for a residence permit?",
  "How do I open a bank account without Schufa?",
  "Wie mache ich eine Anmeldung?",
]

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const [showDebug, setShowDebug] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendQuery = async (question) => {
    if (!question.trim() || loading) return

    const userMsg = { role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const resp = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode: 'full' }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const data = await resp.json()
      const assistantMsg = { role: 'assistant', content: data.answer, data }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the backend is running.`,
        data: null,
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuery(input)
  }

  const totalTime = (timing) => {
    if (!timing) return ''
    return Object.values(timing).reduce((a, b) => a + b, 0).toFixed(1)
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.title}>German Bureaucracy Helper</h1>
          <span style={styles.subtitle}>RAG-powered assistant for international residents</span>
        </div>
        <div style={styles.headerRight}>
          <button
            onClick={() => setShowDebug(d => !d)}
            style={{...styles.debugToggle, ...(showDebug ? styles.debugToggleActive : {})}}
          >
            {showDebug ? '🔍 Debug ON' : '🔍 Debug'}
          </button>
          <div style={{
            ...styles.statusDot,
            backgroundColor: health?.status === 'healthy' ? '#22c55e' : health?.status === 'degraded' ? '#eab308' : '#ef4444'
          }} title={health?.status || 'checking...'} />
        </div>
      </header>

      {/* Messages */}
      <main style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.welcome}>
            <h2 style={styles.welcomeTitle}>Welcome! Ask me anything about German bureaucracy.</h2>
            <p style={styles.welcomeText}>
              I can help with Anmeldung, visa applications, health insurance, taxes, banking, and more.
            </p>
            <div style={styles.suggestions}>
              {SUGGESTED_QUESTIONS.map((q, i) => (
                <button key={i} style={styles.suggestionBtn} onClick={() => sendQuery(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={msg.role === 'user' ? styles.userRow : styles.assistantRow}>
            <div style={msg.role === 'user' ? styles.userBubble : styles.assistantBubble}>
              <div style={styles.messageText}>{msg.content}</div>

              {/* Sources */}
              {msg.data?.sources?.length > 0 && (
                <div style={styles.sources}>
                  <div style={styles.sourcesLabel}>Sources:</div>
                  {msg.data.sources.map((src, j) => (
                    <a key={j} href={src.url} target="_blank" rel="noopener noreferrer" style={styles.sourceLink}>
                      {src.source} — {src.title}
                    </a>
                  ))}
                </div>
              )}

              {/* Debug info */}
              {showDebug && msg.data && (
                <div style={styles.debug}>
                  <div style={styles.debugGrid}>
                    <span style={styles.debugLabel}>Category:</span>
                    <span>{msg.data.intent_category}</span>
                    <span style={styles.debugLabel}>Specificity:</span>
                    <span>{msg.data.intent_specificity}</span>
                    <span style={styles.debugLabel}>Levels:</span>
                    <span>{msg.data.search_levels?.join(', ')}</span>
                    <span style={styles.debugLabel}>Retrieved:</span>
                    <span>{msg.data.retrieved_count} → reranked to {msg.data.reranked_count}</span>
                    <span style={styles.debugLabel}>Chunks used:</span>
                    <span>{msg.data.chunks_used}</span>
                    <span style={styles.debugLabel}>Time:</span>
                    <span>{totalTime(msg.data.timing)}s</span>
                  </div>
                  {msg.data.reformulated_query && msg.data.reformulated_query !== msg.content && (
                    <div style={styles.debugReformulated}>
                      Reformulated: "{msg.data.reformulated_query}"
                    </div>
                  )}
                  {msg.data.expanded_queries?.length > 1 && (
                    <div style={styles.debugExpanded}>
                      Expanded: {msg.data.expanded_queries.length} variants
                    </div>
                  )}
                  {msg.data.timing && (
                    <div style={styles.debugTiming}>
                      {Object.entries(msg.data.timing).map(([k, v]) => (
                        <span key={k} style={styles.timingTag}>{k}: {v.toFixed(1)}s</span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div style={styles.assistantRow}>
            <div style={styles.assistantBubble}>
              <div style={styles.loadingDots}>
                <span style={styles.dot}>●</span>
                <span style={{...styles.dot, animationDelay: '0.2s'}}>●</span>
                <span style={{...styles.dot, animationDelay: '0.4s'}}>●</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <form onSubmit={handleSubmit} style={styles.inputArea}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about Anmeldung, visa, insurance, taxes..."
          style={styles.input}
          disabled={loading}
        />
        <button type="submit" style={styles.sendBtn} disabled={loading || !input.trim()}>
          Send
        </button>
      </form>

      <style>{keyframes}</style>
    </div>
  )
}

const keyframes = `
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }
`

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    maxWidth: 800,
    margin: '0 auto',
    fontFamily: "'Inter', -apple-system, sans-serif",
    color: '#1a1a1a',
    backgroundColor: '#fafafa',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    borderBottom: '1px solid #e5e5e5',
    backgroundColor: '#fff',
  },
  headerLeft: { display: 'flex', flexDirection: 'column', gap: 2 },
  headerRight: { display: 'flex', alignItems: 'center', gap: 12 },
  title: { margin: 0, fontSize: 18, fontWeight: 600, color: '#111' },
  subtitle: { fontSize: 13, color: '#888' },
  statusDot: { width: 10, height: 10, borderRadius: '50%' },
  debugToggle: {
    padding: '4px 10px', fontSize: 12, border: '1px solid #ddd',
    borderRadius: 6, background: '#fff', cursor: 'pointer', color: '#666',
  },
  debugToggleActive: { background: '#f0f4ff', borderColor: '#93b4f8', color: '#2563eb' },

  messages: {
    flex: 1, overflowY: 'auto', padding: '24px 16px',
    display: 'flex', flexDirection: 'column', gap: 16,
  },

  welcome: { textAlign: 'center', padding: '60px 20px' },
  welcomeTitle: { fontSize: 22, fontWeight: 600, marginBottom: 8, color: '#111' },
  welcomeText: { fontSize: 15, color: '#666', marginBottom: 28 },
  suggestions: {
    display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center',
  },
  suggestionBtn: {
    padding: '8px 14px', fontSize: 13, border: '1px solid #ddd',
    borderRadius: 8, background: '#fff', cursor: 'pointer', color: '#333',
    transition: 'all 0.15s',
    maxWidth: 280, textAlign: 'left',
  },

  userRow: { display: 'flex', justifyContent: 'flex-end' },
  assistantRow: { display: 'flex', justifyContent: 'flex-start' },
  userBubble: {
    maxWidth: '75%', padding: '10px 16px', borderRadius: '16px 16px 4px 16px',
    backgroundColor: '#2563eb', color: '#fff', fontSize: 14, lineHeight: 1.6,
  },
  assistantBubble: {
    maxWidth: '85%', padding: '14px 18px', borderRadius: '16px 16px 16px 4px',
    backgroundColor: '#fff', border: '1px solid #e5e5e5', fontSize: 14, lineHeight: 1.7,
  },
  messageText: { whiteSpace: 'pre-wrap' },

  sources: {
    marginTop: 12, paddingTop: 10, borderTop: '1px solid #f0f0f0',
    display: 'flex', flexDirection: 'column', gap: 4,
  },
  sourcesLabel: { fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: 0.5 },
  sourceLink: {
    fontSize: 12, color: '#2563eb', textDecoration: 'none',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },

  debug: {
    marginTop: 12, padding: 10, backgroundColor: '#f8f9fc',
    borderRadius: 8, fontSize: 12, color: '#555',
  },
  debugGrid: {
    display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 10px',
  },
  debugLabel: { fontWeight: 500, color: '#888' },
  debugReformulated: { marginTop: 6, fontStyle: 'italic', color: '#666' },
  debugExpanded: { marginTop: 2, color: '#666' },
  debugTiming: { marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 },
  timingTag: {
    padding: '2px 6px', backgroundColor: '#e8ecf4', borderRadius: 4, fontSize: 11, color: '#555',
  },

  loadingDots: { display: 'flex', gap: 4, padding: '4px 0' },
  dot: { animation: 'pulse 1s ease-in-out infinite', fontSize: 12, color: '#999' },

  inputArea: {
    display: 'flex', gap: 8, padding: '12px 16px',
    borderTop: '1px solid #e5e5e5', backgroundColor: '#fff',
  },
  input: {
    flex: 1, padding: '10px 14px', fontSize: 14, border: '1px solid #ddd',
    borderRadius: 10, outline: 'none', fontFamily: 'inherit',
  },
  sendBtn: {
    padding: '10px 20px', fontSize: 14, fontWeight: 500,
    border: 'none', borderRadius: 10, backgroundColor: '#2563eb',
    color: '#fff', cursor: 'pointer', fontFamily: 'inherit',
  },
}