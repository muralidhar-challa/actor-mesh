import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface ActorInfo { id: string; inbox: number; outbox: number; last_seen: number; alive: boolean }
interface MessageEntry { topic: string; origin: string; payload_preview: string; timestamp: number }
interface MeshState { actors: Record<string, ActorInfo>; messages: MessageEntry[]; connected: boolean }

function fmtBytes(n: number): string {
  if (n === 0) return "0"; if (n < 1024) return n + " B"; return (n / 1024).toFixed(1) + " K";
}

export default function Dashboard() {
  const [state, setState] = useState<MeshState>({ actors: {}, messages: [], connected: false });

  const refresh = useCallback(async () => {
    try { setState(await invoke<MeshState>("get_state")); } catch(e) {}
  }, []);

  useEffect(() => {
    refresh();
    const unlisten = listen("mesh-update", () => refresh());
    return () => { unlisten.then(f => f()); };
  }, [refresh]);

  const alive = Object.values(state.actors).filter(a => a.alive).length;
  const ids = Object.keys(state.actors).sort();

  return (
    <div>
      <div className="stat-row">
        <div className="stat"><div className="stat-num">{alive}</div><div className="stat-lbl">Actors</div></div>
        <div className="stat"><div className="stat-num">{state.messages.length}</div><div className="stat-lbl">Messages</div></div>
        <div className="stat"><div className="stat-num">{state.connected ? "●" : "○"}</div><div className="stat-lbl">Connected</div></div>
      </div>

      <h2>Actors</h2>
      <table><thead><tr><th>ID</th><th>Inbox</th><th>Outbox</th><th>Status</th></tr></thead>
        <tbody>
          {ids.length === 0
            ? <tr><td colSpan={4} style={{textAlign:"center",color:"var(--muted)"}}>no actors yet — start the mesh</td></tr>
            : ids.map(id => {
                const a = state.actors[id];
                return <tr key={id}>
                  <td><span className={`dot ${a.alive?"dot-alive":"dot-dead"}`}/>{id}</td>
                  <td>{fmtBytes(a.inbox)}</td><td>{fmtBytes(a.outbox)}</td>
                  <td>{a.alive ? <span className="pill">alive</span> : <span className="pill" style={{opacity:0.5}}>offline</span>}</td>
                </tr>;
              })
          }
        </tbody>
      </table>

      <h2>Live Messages</h2>
      <div style={{maxHeight:300,overflowY:"auto",border:"1px solid var(--border)",background:"var(--card-bg)"}}>
        {state.messages.length === 0
          ? <div style={{padding:"1rem",color:"var(--muted)",textAlign:"center"}}>waiting...</div>
          : state.messages.slice(-100).reverse().map((m,i) => {
              const d = new Date(m.timestamp*1000);
              return <div className="msg-row" key={i}>
                <span style={{fontWeight:600,color:"var(--accent)"}}>{m.topic}</span>
                <span style={{color:"var(--muted)"}}>{m.origin}</span>
                <span style={{fontFamily:"var(--mono)"}}>{m.payload_preview}</span>
                <span style={{color:"var(--muted)",textAlign:"right"}}>{d.toLocaleTimeString()}</span>
              </div>;
            })
        }
      </div>

      <div className="status-bar">
        <span style={{color:state.connected?"#22c55e":"var(--muted)"}}>{state.connected?"connected":"disconnected"}</span>
      </div>
    </div>
  );
}
