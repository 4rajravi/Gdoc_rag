import { useState, useRef, useEffect } from 'react'

const API_URL = '/api'

const CATEGORIES = [
  { id: 'anmeldung', label: 'Anmeldung', icon: '📍' },
  { id: 'visa', label: 'Visa & Permits', icon: '🛂' },
  { id: 'tax', label: 'Taxes', icon: '📊' },
  { id: 'health_insurance', label: 'Insurance', icon: '🏥' },
  { id: 'banking', label: 'Banking', icon: '🏦' },
  { id: 'housing', label: 'Housing', icon: '🏠' },
  { id: 'work', label: 'Work', icon: '💼' },
  { id: 'university', label: 'University', icon: '🎓' },
]

const STARTERS = [
  "How do I register my address after moving?",
  "What health insurance do I need as a student?",
  "How do I file my first Steuererklärung?",
  "What's the process for a Blue Card?",
  "How do I open a bank account without Schufa?",
  "Wie mache ich eine Anmeldung in Hamburg?",
]

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const [showDebug, setShowDebug] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/health`).then(r => r.json()).then(setHealth).catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async (q) => {
    if (!q.trim() || loading) return
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setInput('')
    setLoading(true)
    try {
      const r = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, mode: 'full' }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer, data }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection error. Is the backend running?', data: null }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const totalTime = (t) => t ? Object.values(t).reduce((a, b) => a + b, 0).toFixed(1) : '—'

  return (
    <>
      <style>{globalStyles}</style>
      <div className="app">
        <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sidebar-header">
            <div className="logo">
              <span className="logo-icon">DE</span>
              <div className="logo-text">
                <span className="logo-title">Bureaucracy</span>
                <span className="logo-sub">RAG Assistant</span>
              </div>
            </div>
          </div>
          <button className="new-chat" onClick={() => setMessages([])}><span>+</span> New conversation</button>
          <div className="sidebar-section">
            <div className="sidebar-label">Topics</div>
            {CATEGORIES.map(c => (
              <button key={c.id} className="topic-btn" onClick={() => send(`Tell me about ${c.label} in Germany`)}>
                <span className="topic-icon">{c.icon}</span>{c.label}
              </button>
            ))}
          </div>
          <div className="sidebar-footer">
            <label className="toggle-row">
              <span>Debug</span>
              <div className={`toggle ${showDebug ? 'on' : ''}`} onClick={() => setShowDebug(d => !d)}><div className="toggle-thumb" /></div>
            </label>
            <div className="status-row">
              <div className={`status-dot ${health?.status === 'healthy' ? 'green' : 'red'}`} />
              <span>{health?.status === 'healthy' ? 'All systems operational' : 'Checking services...'}</span>
            </div>
          </div>
        </aside>

        <main className="chat-area">
          <div className="chat-header">
            <button className="menu-btn" onClick={() => setSidebarOpen(s => !s)}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
            </button>
            <div className="chat-header-info">
              <span className="chat-header-title">German Bureaucracy Helper</span>
              <span className="chat-header-sub">multilingual-e5 · Qdrant · cross-encoder reranker · Ollama</span>
            </div>
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-hero">
                  <div className="empty-badge">RAG-powered</div>
                  <h1>Navigate German<br/>bureaucracy with ease</h1>
                  <p>Ask about Anmeldung, visa, insurance, taxes — in English or German.</p>
                </div>
                <div className="starters">
                  {STARTERS.map((q, i) => (
                    <button key={i} className="starter-card" onClick={() => send(q)} style={{ animationDelay: `${i * 0.06}s` }}>
                      <span className="starter-arrow">→</span>{q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`msg-row ${msg.role}`}>
                  {msg.role === 'assistant' && <div className="avatar">DE</div>}
                  <div className={`msg-bubble ${msg.role}`}>
                    <div className="msg-text">{msg.content}</div>
                    {msg.data?.sources?.length > 0 && (
                      <div className="sources">
                        <span className="sources-label">Sources</span>
                        {msg.data.sources.map((s, j) => (
                          <a key={j} href={s.url} target="_blank" rel="noopener noreferrer" className="source-chip">
                            {s.source}
                            <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
                          </a>
                        ))}
                      </div>
                    )}
                    {showDebug && msg.data && (
                      <div className="debug-panel">
                        <div className="debug-row">
                          <span className="debug-tag">{msg.data.intent_category}</span>
                          <span className="debug-tag">{msg.data.intent_specificity}</span>
                          <span className="debug-tag">{msg.data.search_levels?.join(' ')}</span>
                          <span className="debug-tag">{msg.data.retrieved_count}→{msg.data.reranked_count} chunks</span>
                          <span className="debug-tag accent">{totalTime(msg.data.timing)}s</span>
                        </div>
                        {msg.data.timing && (
                          <div className="debug-timing">
                            {Object.entries(msg.data.timing).map(([k, v]) => (
                              <div key={k} className="timing-bar-row">
                                <span className="timing-label">{k}</span>
                                <div className="timing-track">
                                  <div className="timing-fill" style={{ width: `${Math.min((v / Math.max(...Object.values(msg.data.timing))) * 100, 100)}%` }} />
                                </div>
                                <span className="timing-val">{v.toFixed(1)}s</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="msg-row assistant">
                <div className="avatar">DE</div>
                <div className="msg-bubble assistant"><div className="typing"><span /><span /><span /></div></div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form className="input-area" onSubmit={e => { e.preventDefault(); send(input) }}>
            <div className="input-wrap">
              <input ref={inputRef} value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about Anmeldung, visa, taxes, insurance..." disabled={loading} />
              <button type="submit" disabled={loading || !input.trim()} className="send-btn">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2Z"/></svg>
              </button>
            </div>
          </form>
        </main>
      </div>
    </>
  )
}

const globalStyles = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  :root {
    --bg-primary: #0a0a0c; --bg-secondary: #111114; --bg-tertiary: #18181c;
    --bg-hover: #1f1f24; --border: #2a2a30; --border-subtle: #1e1e24;
    --text-primary: #e8e8ec; --text-secondary: #8e8e96; --text-tertiary: #5a5a62;
    --accent: #6366f1; --accent-dim: #4f46e520; --accent-glow: #6366f130;
    --user-bg: #1a1a40; --user-border: #2d2d5e;
    --green: #22c55e; --red: #ef4444;
    --font: 'DM Sans', -apple-system, sans-serif; --mono: 'JetBrains Mono', monospace;
  }
  body { font-family: var(--font); background: var(--bg-primary); color: var(--text-primary); -webkit-font-smoothing: antialiased; }
  .app { display: flex; height: 100vh; overflow: hidden; }
  .sidebar { width: 260px; background: var(--bg-secondary); border-right: 1px solid var(--border-subtle); display: flex; flex-direction: column; transition: width 0.2s; overflow: hidden; flex-shrink: 0; }
  .sidebar.collapsed { width: 0; border: none; }
  .sidebar-header { padding: 20px 16px 12px; }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-icon { width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, var(--accent), #818cf8); display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: #fff; letter-spacing: 0.5px; }
  .logo-text { display: flex; flex-direction: column; }
  .logo-title { font-size: 14px; font-weight: 600; color: var(--text-primary); }
  .logo-sub { font-size: 11px; color: var(--text-tertiary); }
  .new-chat { margin: 8px 12px; padding: 9px 14px; border-radius: 8px; border: 1px dashed var(--border); background: transparent; color: var(--text-secondary); font: 13px var(--font); cursor: pointer; display: flex; align-items: center; gap: 8px; }
  .new-chat:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
  .new-chat span { font-size: 16px; }
  .sidebar-section { flex: 1; overflow-y: auto; padding: 12px; }
  .sidebar-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-tertiary); padding: 8px 6px 6px; }
  .topic-btn { width: 100%; padding: 8px 10px; border-radius: 6px; border: none; background: transparent; color: var(--text-secondary); font: 13px var(--font); cursor: pointer; display: flex; align-items: center; gap: 8px; text-align: left; }
  .topic-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
  .topic-icon { font-size: 14px; }
  .sidebar-footer { padding: 12px 16px; border-top: 1px solid var(--border-subtle); }
  .toggle-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--text-secondary); cursor: pointer; }
  .toggle { width: 34px; height: 18px; border-radius: 10px; background: var(--bg-hover); border: 1px solid var(--border); position: relative; cursor: pointer; transition: all 0.2s; }
  .toggle.on { background: var(--accent); border-color: var(--accent); }
  .toggle-thumb { position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; border-radius: 50%; background: #fff; transition: transform 0.2s; }
  .toggle.on .toggle-thumb { transform: translateX(16px); }
  .status-row { display: flex; align-items: center; gap: 6px; margin-top: 10px; font-size: 11px; color: var(--text-tertiary); }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; }
  .status-dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.red { background: var(--red); }
  .chat-area { flex: 1; display: flex; flex-direction: column; min-width: 0; background: var(--bg-primary); }
  .chat-header { padding: 12px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 14px; background: var(--bg-secondary); }
  .menu-btn { background: none; border: none; color: var(--text-secondary); cursor: pointer; padding: 4px; border-radius: 4px; }
  .menu-btn:hover { color: var(--text-primary); }
  .chat-header-title { font-size: 14px; font-weight: 500; }
  .chat-header-sub { font-size: 11px; color: var(--text-tertiary); font-family: var(--mono); }
  .chat-header-info { display: flex; flex-direction: column; gap: 1px; }
  .messages { flex: 1; overflow-y: auto; padding: 24px 0; scroll-behavior: smooth; }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .msg-row { display: flex; gap: 12px; padding: 6px 28px; max-width: 900px; margin: 0 auto; width: 100%; animation: msgIn 0.25s ease-out; }
  .msg-row.user { justify-content: flex-end; }
  @keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .avatar { width: 28px; height: 28px; border-radius: 6px; flex-shrink: 0; background: linear-gradient(135deg, var(--accent), #818cf8); display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: 700; color: #fff; letter-spacing: 0.5px; margin-top: 4px; }
  .msg-bubble { max-width: 720px; border-radius: 12px; padding: 14px 18px; line-height: 1.7; font-size: 14px; }
  .msg-bubble.user { background: var(--user-bg); border: 1px solid var(--user-border); color: #c7c7ff; }
  .msg-bubble.assistant { background: var(--bg-tertiary); border: 1px solid var(--border-subtle); }
  .msg-text { white-space: pre-wrap; word-break: break-word; }
  .sources { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-subtle); display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .sources-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--text-tertiary); margin-right: 4px; }
  .source-chip { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-family: var(--mono); background: var(--accent-dim); color: #818cf8; text-decoration: none; border: 1px solid #6366f120; }
  .source-chip:hover { background: #6366f130; border-color: var(--accent); }
  .debug-panel { margin-top: 12px; padding: 10px 12px; border-radius: 8px; background: var(--bg-primary); border: 1px solid var(--border-subtle); }
  .debug-row { display: flex; flex-wrap: wrap; gap: 5px; }
  .debug-tag { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-family: var(--mono); background: var(--bg-hover); color: var(--text-secondary); border: 1px solid var(--border-subtle); }
  .debug-tag.accent { background: var(--accent-dim); color: #818cf8; border-color: #6366f130; }
  .debug-timing { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
  .timing-bar-row { display: flex; align-items: center; gap: 8px; }
  .timing-label { font-size: 10px; font-family: var(--mono); color: var(--text-tertiary); width: 80px; text-align: right; }
  .timing-track { flex: 1; height: 4px; background: var(--bg-hover); border-radius: 2px; overflow: hidden; }
  .timing-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.4s ease; }
  .timing-val { font-size: 10px; font-family: var(--mono); color: var(--text-tertiary); width: 36px; }
  .typing { display: flex; gap: 4px; padding: 4px 0; }
  .typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--text-tertiary); animation: blink 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 60%, 100% { opacity: 0.2; } 30% { opacity: 1; } }
  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 40px 28px; }
  .empty-hero { text-align: center; margin-bottom: 40px; }
  .empty-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-family: var(--mono); font-weight: 500; background: var(--accent-dim); color: #818cf8; border: 1px solid #6366f120; margin-bottom: 16px; }
  .empty-hero h1 { font-size: 32px; font-weight: 600; line-height: 1.25; letter-spacing: -0.5px; color: var(--text-primary); margin-bottom: 10px; }
  .empty-hero p { font-size: 15px; color: var(--text-secondary); max-width: 440px; }
  .starters { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; width: 100%; max-width: 640px; }
  .starter-card { padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-secondary); font: 13px var(--font); text-align: left; cursor: pointer; display: flex; align-items: flex-start; gap: 8px; animation: fadeUp 0.3s ease-out both; }
  .starter-card:hover { border-color: var(--accent); color: var(--text-primary); background: var(--bg-tertiary); transform: translateY(-1px); }
  .starter-arrow { color: var(--accent); flex-shrink: 0; margin-top: 1px; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  .input-area { padding: 16px 28px 20px; background: linear-gradient(to top, var(--bg-primary) 80%, transparent); }
  .input-wrap { max-width: 900px; margin: 0 auto; display: flex; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-secondary); overflow: hidden; }
  .input-wrap:focus-within { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
  .input-wrap input { flex: 1; padding: 13px 16px; background: transparent; border: none; color: var(--text-primary); font: 14px var(--font); outline: none; }
  .input-wrap input::placeholder { color: var(--text-tertiary); }
  .send-btn { padding: 8px 14px; background: transparent; border: none; color: var(--text-tertiary); cursor: pointer; display: flex; align-items: center; }
  .send-btn:not(:disabled):hover { color: var(--accent); }
  .send-btn:disabled { opacity: 0.3; cursor: default; }
  @media (max-width: 768px) { .sidebar { position: fixed; z-index: 100; height: 100vh; } .sidebar.collapsed { width: 0; } .empty-hero h1 { font-size: 24px; } .starters { grid-template-columns: 1fr; } }
`