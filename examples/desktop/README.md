# Actor Mesh Desktop Dashboard

A Tauri desktop app that IS an actor on the mesh. Subscribes
to all topics, renders a live dashboard with the same design
language as the Quarto docs (IBM Plex, #f9f9f8, stat cards).

## Architecture

```
┌──────────── Tauri App ──────────────┐
│  src/index.html   ←── IPC ──→  src-tauri/src/main.rs   │
│  (rendering)                    (nng actor + state)     │
└─────────────────────────────────────────────────────────┘
         │ nng sub (heartbeat, *)    │ nng pub (commands)
         ↓                           ↓
    tcp://proxy:5556            tcp://proxy:5557
```

The Rust backend is an nng actor. The frontend is a rendering surface.
Every widget is a topic subscription. Every button is a topic publish.

## Prerequisites

### Linux (Fedora)
```sh
sudo dnf install webkit2gtk4.1-devel gtk3-devel libsoup3-devel \
  javascriptcoregtk6.0-devel cargo rustc nodejs npm
```

### macOS
```sh
brew install nng lmdb
# Xcode CLI tools already include required frameworks
```

### Windows
```sh
# MSYS2
pacman -S mingw-w64-x86_64-nng mingw-w64-x86_64-lmdb
# Also needs WebView2 (built into Windows 10+)
```

## Setup

```sh
cd examples/desktop
npm install
cargo install tauri-cli --version "^2"
```

## Run

```sh
# Start the mesh first
cd ../.. && make && ./mesh-proxy &
# Start your actors (sql-tool, llm-agent, etc.)
# Then launch the dashboard:
cd examples/desktop
cargo tauri dev
```

## Build (standalone binary)

```sh
cargo tauri build
# → src-tauri/target/release/bundle/
```

## Design

Uses the exact same palette and typography as the Quarto docs:

| Token | Value |
|---|---|
| Background | `#f9f9f8` |
| Accent | `#c41e3a` |
| Dark | `#0a0a0a` |
| Border | `#e2e2e2` |
| Sans font | IBM Plex Sans |
| Mono font | IBM Plex Mono |
