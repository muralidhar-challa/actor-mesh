// main.rs — Actor Mesh desktop dashboard
//
// This is an ACTOR on the mesh. It subscribes to all topics,
// maintains live state, and exposes it to the Tauri frontend.
// The frontend is a rendering surface — all mesh logic is here.

use nng::{
    options::{protocol::pubsub::Subscribe, Options, RecvTimeout},
    Protocol, Socket,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use tauri::State;

// ── Mesh state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ActorInfo {
    id: String,
    inbox: u64,
    outbox: u64,
    last_seen: u64,
    alive: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MessageEntry {
    topic: String,
    origin: String,
    payload_preview: String,
    timestamp: u64,
}

#[derive(Debug, Default, Serialize, Clone)]
struct MeshState {
    actors: HashMap<String, ActorInfo>,
    messages: Vec<MessageEntry>,
    connected: bool,
    proxy_url: String,
}

struct AppState {
    mesh: Mutex<MeshState>,
}

// ── Wire format helpers ────────────────────────────────────────────────────

const HEADER_SIZE: usize = 256;

fn read_topic(data: &[u8]) -> String {
    let end = data.iter().position(|&b| b == 0).unwrap_or(32);
    String::from_utf8_lossy(&data[..end.min(32)]).to_string()
}

fn ts_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ── Tauri commands ─────────────────────────────────────────────────────────

#[tauri::command]
fn get_state(state: State<AppState>) -> MeshState {
    state.mesh.lock().unwrap().clone()
}

#[tauri::command]
fn publish_message(topic: String, payload: String) -> Result<(), String> {
    let pub_sock = Socket::new(Protocol::Pub0).map_err(|e| e.to_string())?;
    pub_sock
        .dial("tcp://127.0.0.1:5557")
        .map_err(|e| e.to_string())?;

    // Build header (256 bytes) + payload
    let mut frame = vec![0u8; HEADER_SIZE + payload.len()];
    let tbytes = topic.as_bytes();
    frame[..tbytes.len().min(31)].copy_from_slice(&tbytes[..tbytes.len().min(31)]);
    frame[HEADER_SIZE..].copy_from_slice(payload.as_bytes());

    pub_sock.send(&frame).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Background mesh listener ───────────────────────────────────────────────

fn mesh_listener(state: tauri::AppHandle, proxy_sub: String) {
    std::thread::spawn(move || {
        let sub = match Socket::new(Protocol::Sub0) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[desktop] sub socket: {e}");
                return;
            }
        };

        if let Err(e) = sub.dial(&proxy_sub) {
            eprintln!("[desktop] sub dial {proxy_sub}: {e}");
            return;
        }

        // Subscribe to everything (empty = all topics)
        let all: Vec<u8> = vec![];
        sub.set_opt::<Subscribe>(all).ok();
        sub.set_opt::<RecvTimeout>(Some(Duration::from_millis(200)))
            .ok();

        let mut ring: Vec<MessageEntry> = Vec::new();

        loop {
            match sub.recv() {
                Ok(msg) => {
                    let data = msg.as_ref();
                    if data.len() < HEADER_SIZE {
                        continue;
                    }

                    let topic = read_topic(&data[0..32]);
                    let origin = read_topic(&data[64..96]);

                    let payload_data = &data[HEADER_SIZE..];
                    let preview = String::from_utf8_lossy(
                        &payload_data[..payload_data.len().min(200)],
                    )
                    .to_string();

                    let ts = ts_now();
                    let entry = MessageEntry {
                        topic: topic.clone(),
                        origin: origin.clone(),
                        payload_preview: preview,
                        timestamp: ts,
                    };

                    // Track actor presence from heartbeat payloads
                    if topic == "heartbeat" {
                        if let Ok(parsed) =
                            serde_json::from_slice::<serde_json::Value>(payload_data)
                        {
                            if let (Some(id), Some(inbox), Some(outbox)) = (
                                parsed["id"].as_str(),
                                parsed["inbox"].as_u64(),
                                parsed["outbox"].as_u64(),
                            ) {
                                let s = state.state::<AppState>();
                                let mut mesh = s.mesh.lock().unwrap();
                                mesh.actors.insert(
                                    id.to_string(),
                                    ActorInfo {
                                        id: id.to_string(),
                                        inbox,
                                        outbox,
                                        last_seen: ts,
                                        alive: true,
                                    },
                                );
                                for actor in mesh.actors.values_mut() {
                                    if ts - actor.last_seen > 15 {
                                        actor.alive = false;
                                    }
                                }
                            }
                        }
                    }

                    // Ring buffer: keep last 200 messages
                    ring.push(entry);
                    if ring.len() > 200 {
                        ring.remove(0);
                    }

                    // Push update to frontend
                    {
                        let s = state.state::<AppState>();
                        let mut mesh = s.mesh.lock().unwrap();
                        mesh.messages = ring.clone();
                        mesh.connected = true;
                    }
                    let _ = state.emit("mesh-update", ());
                }
                Err(nng::Error::TimedOut) => {
                    let s = state.state::<AppState>();
                    let mut mesh = s.mesh.lock().unwrap();
                    let now = ts_now();
                    for actor in mesh.actors.values_mut() {
                        if now - actor.last_seen > 15 {
                            actor.alive = false;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
}

// ── Entry ───────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let proxy_sub =
        std::env::var("ACTOR_BUS_SUB").unwrap_or_else(|_| "tcp://127.0.0.1:5556".into());
    let proxy_pub =
        std::env::var("ACTOR_BUS_PUB").unwrap_or_else(|_| "tcp://127.0.0.1:5557".into());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            mesh: Mutex::new(MeshState {
                proxy_url: proxy_pub.clone(),
                ..Default::default()
            }),
        })
        .invoke_handler(tauri::generate_handler![get_state, publish_message])
        .setup(|app| {
            let handle = app.handle().clone();
            mesh_listener(handle, proxy_sub);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
