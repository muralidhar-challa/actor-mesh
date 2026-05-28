import { useEffect, useState, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface Message {
  role: "user" | "agent";
  text: string;
}

export default function Chat() {
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleMeshUpdate = useCallback(async () => {
    if (!thinking) return;
    try {
      const s = await invoke<{ messages: { topic: string; payload_preview: string }[] }>("get_state");
      const responses = s.messages.filter(m => m.topic === "agent_response");
      if (responses.length > 0) {
        const last = responses[responses.length - 1];
        // Decode from base64 (since we can't pass raw bytes through invoke easily)
        // Actually, let's use the payload directly — it's stored as JSON string in mesh state
        try {
          const parsed = JSON.parse(last.payload_preview);
          if (parsed.answer) {
            setHistory(h => [...h, { role: "agent", text: parsed.answer }]);
            setThinking(false);
          }
        } catch {
          // payload_preview might be truncated, try to extract answer
          const match = last.payload_preview.match(/"answer":"([^"]+)"/);
          if (match) {
            setHistory(h => [...h, { role: "agent", text: match[1] }]);
            setThinking(false);
          }
        }
      }
    } catch(e) {}
  }, [thinking]);

  useEffect(() => {
    const unlisten = listen("mesh-update", () => handleMeshUpdate());
    return () => { unlisten.then(f => f()); };
  }, [handleMeshUpdate]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const send = async () => {
    if (!input.trim()) return;
    setHistory(h => [...h, { role: "user", text: input }]);
    setThinking(true);
    try {
      await invoke("chat_send", { query: input });
    } catch(e) {
      alert("send failed: " + e);
    }
    setInput("");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 100px)" }}>
      <h2>Chat</h2>
      <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem 0" }}>
        {history.map((m, i) => (
          <div key={i} style={{
            marginBottom: "0.5rem", padding: "0.5rem 0.8rem",
            background: m.role === "user" ? "var(--card-bg)" : "#f5f5f4",
            borderLeft: `3px solid ${m.role === "user" ? "var(--accent)" : "var(--dark)"}`,
            fontFamily: m.role === "agent" ? "var(--sans)" : "var(--mono)",
            fontSize: "0.85rem",
          }}>
            <strong style={{ color: m.role === "user" ? "var(--accent)" : "var(--dark)", fontSize: "0.7rem", textTransform: "uppercase" }}>
              {m.role}
            </strong>
            <div style={{ marginTop: "0.2rem", color: "var(--fg-dim)" }}>{m.text}</div>
          </div>
        ))}
        {thinking && <div style={{ padding: "0.5rem 0.8rem", color: "var(--muted)", fontStyle: "italic" }}>thinking...</div>}
        <div ref={bottomRef} />
      </div>
      <div className="cmd-bar">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send()} placeholder="ask something..." />
        <button onClick={send}>Send</button>
      </div>
    </div>
  );
}
