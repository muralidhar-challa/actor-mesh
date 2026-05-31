import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import Chat from "./panels/Chat";

export interface ChatMessage { role: "user" | "agent"; text: string }

interface ActorInfo { id: string; inbox: number; outbox: number; alive: boolean }
interface MeshState { actors: Record<string, ActorInfo>; messages: any[]; connected: boolean }

function fmtK(n: number) { return n === 0 ? "0" : (n/1024).toFixed(1) + "K"; }

function MeshTab() {
  const [state, setState] = useState<MeshState>({actors:{},messages:[],connected:false});
  useEffect(() => {
    const i = setInterval(async () => { try { setState(await invoke<MeshState>("get_state")); } catch(e){} }, 1000);
    return () => clearInterval(i);
  }, []);

  const alive = Object.values(state.actors).filter(a=>a.alive).length;
  const ids = Object.keys(state.actors).sort();

  return <div>
    <div className="stat-row">
      <div className="stat"><div className="stat-num">{alive}</div><div className="stat-lbl">Actors</div></div>
      <div className="stat"><div className="stat-num">{state.messages.length}</div><div className="stat-lbl">Messages</div></div>
      <div className="stat"><div className="stat-num">{state.connected?"●":"○"}</div><div className="stat-lbl">Connected</div></div>
    </div>
    <h2>Actors</h2>
    <table><thead><tr><th>ID</th><th>Inbox</th><th>Outbox</th><th>Status</th></tr></thead>
      <tbody>{ids.length===0 ? <tr><td colSpan={4} style={{textAlign:"center",color:"var(--muted)"}}>none</td></tr>
        : ids.map(id => { const a=state.actors[id]; return <tr key={id}>
          <td><span className={`dot ${a.alive?"dot-alive":"dot-dead"}`}/>{id}</td>
          <td>{fmtK(a.inbox)}</td><td>{fmtK(a.outbox)}</td>
          <td>{a.alive?<span className="pill">alive</span>:<span className="pill" style={{opacity:0.5}}>offline</span>}</td>
        </tr>; })}
    </tbody></table>
  </div>;
}

function ActorDot() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const i = setInterval(async () => {
      try { const s = await invoke<MeshState>("get_state");
        setCount(Object.values(s.actors).filter(a=>a.alive).length); } catch(e){}
    }, 2000);
    return () => clearInterval(i);
  }, []);
  const c = count > 0 ? "#22c55e" : "var(--muted)";
  return <span style={{color:c}}>{count > 0 ? `● ${count} actor${count>1?"s":""}` : "○ connecting"}</span>;
}

export default function App() {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [tab, setTab] = useState<"chat"|"mesh">("chat");

  return <div className="app">
    <div className="tabs">
      <button className={`tab ${tab==="chat"?"active":""}`} onClick={()=>setTab("chat")}>Chat</button>
      <button className={`tab ${tab==="mesh"?"active":""}`} onClick={()=>setTab("mesh")}>Mesh</button>
      <div style={{flex:1}}/>
      <span style={{padding:"0.6rem 1rem",fontSize:"0.72rem",fontFamily:"var(--mono)"}}><ActorDot/></span>
    </div>
    <div className="content">
      {tab==="chat" && <Chat history={chatHistory} setHistory={setChatHistory}/>}
      {tab==="mesh" && <MeshTab/>}
    </div>
  </div>;
}
