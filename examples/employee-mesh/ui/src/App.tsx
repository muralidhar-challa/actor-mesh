import { useState } from "react";
import Dashboard from "./panels/Dashboard";
import Chat from "./panels/Chat";

type Tab = "dashboard" | "chat";

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");

  return (
    <div className="app">
      <div className="tabs">
        <button className={`tab ${tab === "dashboard" ? "active" : ""}`} onClick={() => setTab("dashboard")}>
          Dashboard
        </button>
        <button className={`tab ${tab === "chat" ? "active" : ""}`} onClick={() => setTab("chat")}>
          Chat
        </button>
      </div>
      <div className="content">
        {tab === "dashboard" && <Dashboard />}
        {tab === "chat" && <Chat />}
      </div>
    </div>
  );
}
