"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Suggestion questions for the empty state ── */
const SUGGESTIONS = [
  "How does authentication work?",
  "Where is error handling implemented?",
  "Explain the database models",
  "What API endpoints are available?",
  "How is input validation done?",
];

/* ══════════════════════════════════════════════════════════════
   CodeModal — Full code viewer for a citation
   ══════════════════════════════════════════════════════════════ */
function CodeModal({ citation, onClose }) {
  useEffect(() => {
    const handleEsc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  return (
    <div className="code-modal-overlay" onClick={onClose}>
      <div className="code-modal" onClick={(e) => e.stopPropagation()}>
        <div className="code-modal-header">
          <div>
            <div className="code-modal-title">{citation.file_path}</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginTop: 2 }}>
              Lines {citation.start_line}–{citation.end_line} · {citation.node_type.replace(/_/g, " ")}
            </div>
          </div>
          <button className="code-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="code-modal-body">
          <pre><code>{citation.full_code}</code></pre>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   CitationCard — Single source reference
   ══════════════════════════════════════════════════════════════ */
function CitationCard({ citation, onClick }) {
  return (
    <div className="citation-card" onClick={() => onClick(citation)}>
      <div className="citation-index">{citation.index}</div>
      <div className="citation-info">
        <div className="citation-file">{citation.file_path}</div>
        <div className="citation-lines">
          Lines {citation.start_line}–{citation.end_line}
          {citation.rerank_score > 0 && (
            <span style={{ marginLeft: 8, color: "var(--accent-emerald)" }}>
              score: {citation.rerank_score.toFixed(2)}
            </span>
          )}
        </div>
        <div className="citation-snippet">{citation.snippet}</div>
      </div>
      <div className="citation-type-badge">
        {citation.node_type.replace(/_/g, " ")}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   MessageBubble — A single chat message (user or assistant)
   ══════════════════════════════════════════════════════════════ */
function MessageBubble({ message, onCitationClick }) {
  const isUser = message.role === "user";

  /* Simple markdown-like rendering for the answer */
  function renderContent(text) {
    if (!text) return null;

    const parts = text.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\])/g);

    return parts.map((part, i) => {
      /* Fenced code block */
      if (part.startsWith("```")) {
        const code = part.replace(/^```\w*\n?/, "").replace(/```$/, "");
        return (
          <pre key={i}>
            <code>{code}</code>
          </pre>
        );
      }
      /* Inline code */
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={i}>{part.slice(1, -1)}</code>;
      }
      /* Bold */
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      /* Citation reference like [1] */
      if (/^\[\d+\]$/.test(part)) {
        return (
          <span
            key={i}
            style={{
              color: "var(--accent-purple)",
              fontWeight: 600,
              cursor: "pointer",
              fontSize: "0.85em",
            }}
          >
            {part}
          </span>
        );
      }
      /* Normal text — split on newlines to create paragraphs */
      return part.split("\n").map((line, j) => (
        <span key={`${i}-${j}`}>
          {line}
          {j < part.split("\n").length - 1 && <br />}
        </span>
      ));
    });
  }

  return (
    <div className="message">
      <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "👤" : "🔍"}
      </div>
      <div className="message-body">
        <div className="message-header">
          <span className="message-sender">{isUser ? "You" : "CodeLens"}</span>
          <span className="message-time">{message.time}</span>
        </div>
        <div className="message-content">{renderContent(message.content)}</div>

        {/* Citations panel */}
        {message.citations && message.citations.length > 0 && (
          <div className="citations-panel">
            <div className="citations-header">
              <span>📎</span>
              <span>{message.citations.length} Sources Referenced</span>
            </div>
            {message.citations.map((c) => (
              <CitationCard key={c.index} citation={c} onClick={onCitationClick} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   TypingIndicator — Animated dots while waiting for response
   ══════════════════════════════════════════════════════════════ */
function TypingIndicator() {
  return (
    <div className="message">
      <div className="message-avatar assistant">🔍</div>
      <div className="message-body">
        <div className="message-header">
          <span className="message-sender">CodeLens</span>
        </div>
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Main Page Component
   ══════════════════════════════════════════════════════════════ */
export default function Home() {
  /* ── State ── */
  const [phase, setPhase] = useState("index"); // "index" | "chat"
  const [repoUrl, setRepoUrl] = useState("");
  const [repoId, setRepoId] = useState("");
  const [indexStatus, setIndexStatus] = useState(null); // null | {status, progress, ...}
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [codeModal, setCodeModal] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const pollRef = useRef(null);

  /* Auto-scroll to latest message */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  /* Focus input on phase change */
  useEffect(() => {
    if (phase === "chat") {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [phase]);

  /* Cleanup polling on unmount */
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  /* ── Index a repository ── */
  const handleIndex = async (e) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setIndexStatus({ status: "queued", progress: 0 });

    try {
      const res = await fetch(`${API_BASE}/api/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repoUrl.trim() }),
      });

      if (!res.ok) throw new Error("Failed to start indexing");

      const data = await res.json();
      setRepoId(data.repo_id);

      /* Poll for status updates */
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/api/index/${data.job_id}`);
          const statusData = await statusRes.json();
          setIndexStatus(statusData);

          if (statusData.status === "complete" || statusData.status === "error") {
            clearInterval(pollRef.current);
            pollRef.current = null;

            if (statusData.status === "complete") {
              setTimeout(() => setPhase("chat"), 1500);
            }
          }
        } catch {
          /* ignore poll errors */
        }
      }, 2000);
    } catch (err) {
      setIndexStatus({ status: "error", error: err.message });
    }
  };

  /* ── Send a question ── */
  const handleSend = async (questionText) => {
    const question = (questionText || input).trim();
    if (!question || isLoading) return;

    const now = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    /* Add user message */
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question, time: now },
    ]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_id: repoId,
          question: question,
          top_k: 5,
        }),
      });

      if (!res.ok) throw new Error("Query failed");

      const data = await res.json();

      const responseTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          time: responseTime,
          model: data.model_used,
          chunks: data.chunks_retrieved,
        },
      ]);
    } catch (err) {
      const errorTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I encountered an error: ${err.message}. Make sure the backend is running at ${API_BASE} and the repository is indexed.`,
          time: errorTime,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ═════════════════════════════════════════════════════════════
     RENDER
     ═════════════════════════════════════════════════════════════ */
  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">⟨/⟩</div>
            <span>Code<span className="logo-text-gradient">Lens</span></span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-md)" }}>
            {phase === "chat" && (
              <div className="repo-pill">
                <span>📦</span>
                <span>{repoId}</span>
              </div>
            )}
            <div className="header-badge">
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent-emerald)", display: "inline-block" }} />
              RAG-Powered
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="main-content">
        {/* ════ INDEX PHASE ════ */}
        {phase === "index" && (
          <div className="glass-card index-section">
            <h1 className="index-title">
              Understand any codebase
              <br />
              <span className="logo-text-gradient">in seconds.</span>
            </h1>
            <p className="index-subtitle">
              Paste a GitHub repository URL and ask questions with natural language.
              Get accurate answers with exact file and line citations.
            </p>

            <form className="index-form" onSubmit={handleIndex}>
              <input
                id="repo-url-input"
                type="url"
                className="index-input"
                placeholder="https://github.com/user/repo"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                disabled={indexStatus && indexStatus.status !== "error"}
              />
              <button
                id="index-btn"
                type="submit"
                className="btn btn-primary"
                disabled={!repoUrl.trim() || (indexStatus && indexStatus.status !== "error")}
              >
                {indexStatus && indexStatus.status !== "error" ? (
                  <span className="loading-spinner" />
                ) : (
                  "Index →"
                )}
              </button>
            </form>

            {/* Progress */}
            {indexStatus && indexStatus.status !== "error" && (
              <div className="progress-container">
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${indexStatus.progress || 0}%` }}
                  />
                </div>
                <div className="progress-status">
                  <span className="progress-label">
                    {indexStatus.status === "queued" && "⏳ Queued..."}
                    {indexStatus.status === "cloning" && "📥 Cloning repository..."}
                    {indexStatus.status === "indexing" && "🧠 Parsing & embedding chunks..."}
                    {indexStatus.status === "complete" && "✅ Ready!"}
                  </span>
                  <span className="progress-percent">{indexStatus.progress || 0}%</span>
                </div>
              </div>
            )}

            {/* Success */}
            {indexStatus && indexStatus.status === "complete" && (
              <div className="success-banner">
                <div className="success-icon">✅</div>
                <div className="success-text">
                  <div className="success-title">Repository indexed successfully</div>
                  <div className="success-detail">
                    {indexStatus.total_chunks} code chunks extracted and embedded.
                    Redirecting to chat...
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {indexStatus && indexStatus.status === "error" && (
              <div
                className="success-banner"
                style={{
                  background: "rgba(244, 63, 94, 0.1)",
                  borderColor: "rgba(244, 63, 94, 0.2)",
                  marginTop: "var(--space-lg)",
                }}
              >
                <div className="success-icon">❌</div>
                <div className="success-text">
                  <div className="success-title" style={{ color: "var(--accent-rose)" }}>
                    Indexing failed
                  </div>
                  <div className="success-detail">{indexStatus.error}</div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ════ CHAT PHASE ════ */}
        {phase === "chat" && (
          <div className="glass-card chat-container">
            <div className="chat-messages">
              {messages.length === 0 && !isLoading && (
                <div className="empty-state">
                  <div className="empty-icon">🔍</div>
                  <div className="empty-title">Ask anything about {repoId}</div>
                  <div className="empty-description">
                    Ask about functions, architecture, patterns, or any part of the codebase.
                    I&apos;ll retrieve the relevant code and answer with exact citations.
                  </div>
                  <div className="suggestion-chips">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        className="suggestion-chip"
                        onClick={() => handleSend(s)}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <MessageBubble
                  key={i}
                  message={msg}
                  onCitationClick={setCodeModal}
                />
              ))}

              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chat-input-container">
              <form className="chat-input-form" onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
                <div className="chat-input-wrapper">
                  <textarea
                    ref={inputRef}
                    id="chat-input"
                    className="chat-input"
                    placeholder={`Ask about ${repoId}...`}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={isLoading}
                  />
                </div>
                <button
                  id="send-btn"
                  type="submit"
                  className="btn btn-primary btn-send"
                  disabled={!input.trim() || isLoading}
                >
                  ↑
                </button>
              </form>
            </div>
          </div>
        )}
      </main>

      {/* ── Code Viewer Modal ── */}
      {codeModal && (
        <CodeModal citation={codeModal} onClose={() => setCodeModal(null)} />
      )}
    </div>
  );
}
