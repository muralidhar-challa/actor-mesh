import { useEffect, useRef, useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { marked } from "marked";
import type { ChatMessage } from "../App";

interface Props { history: ChatMessage[]; setHistory: (h: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void }

export default function Chat({ history, setHistory }: Props) {
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleMeshUpdate = useCallback(async () => {
    if (!thinking) return;
    try {
      const s = await invoke<{ messages: { topic: string; answer: string }[] }>("get_state");
      const responses = s.messages.filter(m => m.topic === "agent_response" && m.answer);
      if (responses.length > 0) {
        setHistory(prev => [...prev, { role: "agent", text: responses[responses.length - 1].answer }]);
        setThinking(false);
      }
    } catch(e) {}
  }, [thinking, setHistory]);

  useEffect(() => {
    if (!thinking) return;
    const interval = setInterval(handleMeshUpdate, 500);
    return () => clearInterval(interval);
  }, [handleMeshUpdate, thinking]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [history]);

  const send = async () => {
    if (!input.trim()) return;
    setHistory(prev => [...prev, { role: "user", text: input }]);
    setThinking(true);
    try { await invoke("chat_send", { query: input }); } catch(e) { alert("send failed: " + e); }
    setInput("");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 100px)" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: "1rem 0" }}>
        {history.map((m, i) =>
          m.role === "user" ? (
            <div key={i} style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.6rem" }}>
              <div style={{
                background: "var(--dark)", color: "#fff", padding: "0.5rem 0.9rem",
                borderRadius: "12px 12px 2px 12px", maxWidth: "70%",
                fontFamily: "var(--sans)", fontSize: "0.88rem", lineHeight: 1.5,
              }}>
                {m.text}
              </div>
            </div>
          ) : (
            <div key={i} style={{ marginBottom: "1rem", fontSize: "0.9rem", lineHeight: 1.7, color: "var(--fg-dim)" }}>
              <div dangerouslySetInnerHTML={{ __html: marked.parse(m.text) as string }} />
            </div>
          )
        )}
        {thinking && <div style={{ color: "var(--muted)", fontStyle: "italic", padding: "0.5rem 0" }}>thinking...</div>}
        <div ref={bottomRef} />
      </div>
      <div className="cmd-bar">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && send()}
          placeholder="ask about employees..."
          style={{
            background: "var(--dark)", color: "#fff", border: "none",
            padding: "0.6rem 1rem", borderRadius: "12px",
            fontFamily: "var(--sans)", fontSize: "0.88rem",
          }}
        />
        <button onClick={send} style={{ borderRadius: "12px", padding: "0.6rem 1.4rem" }}>Send</button>
      </div>
    </div>
  );
}
