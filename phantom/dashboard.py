#!/usr/bin/env python3
"""
Phantom Memory Dashboard — Real-time visualization of the memory system.

Usage:
    from phantom.dashboard import create_app
    from aiohttp import web
    app = create_app(vault_path="~/my_vault", db_path="~/phantom_db")
    web.run_app(app, port=8422)

Or via CLI:
    phantom start  # starts daemon + dashboard on port 8422
"""

import os
import re
from pathlib import Path

from aiohttp import web


# ── Dashboard HTML ──

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phantom Memory — Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0e14;
    --surface: #111820;
    --surface-2: #1a2230;
    --border: #1e2a3a;
    --text: #c5d0dc;
    --text-dim: #5c6a7a;
    --cyan: #00e5ff;
    --green: #00ff9d;
    --amber: #ffb300;
    --red: #ff3d71;
    --purple: #b388ff;
    --blue: #448aff;
    --cyan-dim: rgba(0, 229, 255, 0.15);
    --green-dim: rgba(0, 255, 157, 0.15);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    overflow: hidden;
    height: 100vh;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px; font-weight: 700; color: var(--cyan);
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
  }
  .status-badge {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 500;
    background: var(--green-dim); color: var(--green);
    border: 1px solid rgba(0, 255, 157, 0.3);
  }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .header-stats {
    display: flex; gap: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }
  .stat { color: var(--text-dim); }
  .stat-value { color: var(--cyan); font-weight: 600; }

  .main {
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    height: calc(100vh - 49px); gap: 1px; background: var(--border);
  }

  .panel { background: var(--surface); overflow-y: auto; }
  .panel-header {
    position: sticky; top: 0; z-index: 10;
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    background: var(--surface-2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .panel-count {
    background: var(--cyan-dim); color: var(--cyan);
    padding: 2px 8px; border-radius: 10px; font-size: 10px;
  }

  .entity-list { padding: 6px; }
  .entity-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 10px; margin: 2px 0; border-radius: 6px;
    cursor: pointer; transition: all 0.15s; border: 1px solid transparent;
  }
  .entity-item:hover { background: var(--surface-2); border-color: var(--border); }
  .entity-item.selected { background: var(--cyan-dim); border-color: rgba(0, 229, 255, 0.3); }
  .entity-name { font-weight: 500; font-size: 12px; }
  .entity-count {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--text-dim); background: var(--surface-2);
    padding: 2px 6px; border-radius: 8px;
  }
  .entity-icon { margin-right: 8px; font-size: 14px; }

  .feed { padding: 10px; }
  .feed-item {
    padding: 10px 12px; margin: 6px 0; border-radius: 8px;
    background: var(--surface-2); border-left: 3px solid var(--border);
    animation: slideIn 0.3s ease-out;
  }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; } }
  .feed-item.type-decision { border-left-color: var(--green); }
  .feed-item.type-task { border-left-color: var(--amber); }
  .feed-item.type-preference { border-left-color: var(--purple); }
  .feed-item.type-quantitative { border-left-color: var(--cyan); }
  .feed-item.type-general { border-left-color: var(--text-dim); }
  .feed-item.superseded { opacity: 0.4; border-left-style: dashed; }
  .feed-item.superseded .feed-text { text-decoration: line-through; }
  .superseded-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    padding: 1px 5px; border-radius: 3px;
    background: rgba(255, 61, 113, 0.15); color: var(--red); margin-left: 6px;
  }
  .feed-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .feed-type {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 6px; border-radius: 4px;
  }
  .feed-type.decision { background: var(--green-dim); color: var(--green); }
  .feed-type.task { background: rgba(255, 179, 0, 0.15); color: var(--amber); }
  .feed-type.preference { background: rgba(179, 136, 255, 0.15); color: var(--purple); }
  .feed-type.quantitative { background: var(--cyan-dim); color: var(--cyan); }
  .feed-type.general { background: rgba(92, 106, 122, 0.2); color: var(--text-dim); }
  .feed-time { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); }
  .feed-text { font-size: 12px; line-height: 1.5; }
  .feed-entities { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
  .feed-entity-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    padding: 2px 6px; border-radius: 4px;
    background: var(--cyan-dim); color: var(--cyan);
    border: 1px solid rgba(0, 229, 255, 0.2);
  }
  .feed-source { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); margin-left: 8px; }

  .stats-section { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  .stats-section-title {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 10px;
  }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-card { background: var(--surface-2); border-radius: 8px; padding: 10px 12px; border: 1px solid var(--border); }
  .stat-card-value { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: var(--cyan); }
  .stat-card-label { font-size: 10px; color: var(--text-dim); margin-top: 2px; }

  .type-bar { display: flex; align-items: center; margin: 4px 0; gap: 8px; }
  .type-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); width: 90px; text-align: right; }
  .type-bar-track { flex: 1; height: 6px; background: var(--surface); border-radius: 3px; overflow: hidden; }
  .type-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  .type-bar-count { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); width: 20px; }

  .recall-item { padding: 8px 10px; margin: 4px 0; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--border); }
  .recall-query { font-size: 11px; color: var(--blue); font-weight: 500; }
  .recall-results { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); margin-top: 3px; }

  .empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 200px; color: var(--text-dim); font-size: 12px; text-align: center; gap: 8px;
  }
  .empty-icon { font-size: 32px; opacity: 0.3; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">◈ PHANTOM MEMORY</div>
    <div class="status-badge"><div class="status-dot"></div><span id="status-text">CONNECTED</span></div>
  </div>
  <div class="header-stats">
    <span class="stat">session <span class="stat-value" id="session-id">—</span></span>
    <span class="stat">uptime <span class="stat-value" id="uptime">0s</span></span>
    <span class="stat">last activity <span class="stat-value" id="last-activity">—</span></span>
  </div>
</div>

<div class="main">
  <div class="panel" id="entity-panel">
    <div class="panel-header">Knowledge Graph <span class="panel-count" id="entity-count">0</span></div>
    <div class="entity-list" id="entity-list"></div>
  </div>

  <div class="panel" id="feed-panel">
    <div class="panel-header">Activity Feed <span class="panel-count" id="feed-count">0</span></div>
    <div class="feed" id="feed"></div>
  </div>

  <div class="panel" id="stats-panel">
    <div class="panel-header">System Stats</div>
    <div class="stats-section">
      <div class="stats-section-title">Memory Store</div>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-card-value" id="total-memories">0</div><div class="stat-card-label">Total Memories</div></div>
        <div class="stat-card"><div class="stat-card-value" id="total-entities">0</div><div class="stat-card-label">Entities</div></div>
        <div class="stat-card"><div class="stat-card-value" id="total-ingested">0</div><div class="stat-card-label">Turns Ingested</div></div>
        <div class="stat-card"><div class="stat-card-value" id="total-deduped">0</div><div class="stat-card-label">Deduped</div></div>
        <div class="stat-card" style="grid-column: span 2;"><div class="stat-card-value" id="total-superseded" style="color: var(--red);">0</div><div class="stat-card-label">Superseded (contradictions resolved)</div></div>
      </div>
    </div>
    <div class="stats-section"><div class="stats-section-title">Fact Types</div><div id="type-breakdown"></div></div>
    <div class="stats-section"><div class="stats-section-title">Recent Decisions</div><div id="decisions-list"></div></div>
    <div class="stats-section"><div class="stats-section-title">Open Tasks</div><div id="tasks-list"></div></div>
  </div>
</div>

<script>
const POLL_INTERVAL = 2000;
let startTime = Date.now();
let lastDataHash = '';

async function fetchData() {
  try {
    const [statsRes, entitiesRes, decisionsRes, tasksRes, feedRes] = await Promise.all([
      fetch('/api/stats'), fetch('/api/entities'), fetch('/api/decisions'),
      fetch('/api/tasks'), fetch('/api/feed'),
    ]);
    const stats = await statsRes.json();
    const entities = await entitiesRes.json();
    const decisions = await decisionsRes.json();
    const tasks = await tasksRes.json();
    const feed = await feedRes.json();

    const dataHash = JSON.stringify({stats, entities: entities.count, feed: feed.length});
    if (dataHash === lastDataHash) return;
    lastDataHash = dataHash;

    updateStats(stats); updateEntities(entities); updateFeed(feed);
    updateDecisions(decisions); updateTasks(tasks); updateTypeBreakdown(feed);
    document.getElementById('last-activity').textContent = new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status-text').textContent = 'DISCONNECTED';
    document.querySelector('.status-badge').style.borderColor = 'rgba(255, 61, 113, 0.3)';
    document.querySelector('.status-badge').style.background = 'rgba(255, 61, 113, 0.15)';
    document.querySelector('.status-badge').style.color = 'var(--red)';
    document.querySelector('.status-dot').style.background = 'var(--red)';
  }
}

function updateStats(stats) {
  document.getElementById('session-id').textContent = stats.session_id || '—';
  document.getElementById('total-memories').textContent = stats.total_memories || 0;
  document.getElementById('total-ingested').textContent = stats.ingested_turns || 0;
  document.getElementById('total-deduped').textContent = stats.deduped_facts || 0;
  document.getElementById('total-superseded').textContent = stats.superseded_facts || 0;
}

function updateEntities(data) {
  const list = document.getElementById('entity-list');
  const entities = data.entities || [];
  document.getElementById('entity-count').textContent = entities.length;
  document.getElementById('total-entities').textContent = entities.length;
  const icons = {'counterparty':'🏢','isda':'📄','csa':'📄','section':'§','cross':'⚡','eligible':'✅','haircut':'✂️','us treasuries':'🏛️','credit':'💳','netting':'🔗','valuation':'📅','minimum':'📊','bbb':'⭐','agency':'🏛️','government':'🏛️'};
  list.innerHTML = entities.map(e => {
    const nameLower = e.name.toLowerCase();
    let icon = '◈';
    for (const [key, val] of Object.entries(icons)) { if (nameLower.includes(key)) { icon = val; break; } }
    return `<div class="entity-item" onclick="this.classList.toggle('selected')"><span><span class="entity-icon">${icon}</span><span class="entity-name">${e.name}</span></span><span class="entity-count">${e.fact_count}</span></div>`;
  }).join('');
}

function updateFeed(facts) {
  const container = document.getElementById('feed');
  document.getElementById('feed-count').textContent = facts.length;
  if (facts.length === 0) { container.innerHTML = '<div class="empty-state"><div class="empty-icon">◈</div>Waiting for conversation data...</div>'; return; }
  container.innerHTML = facts.slice().reverse().map(f => {
    const type = f.type || 'general';
    const entities = f.entities || [];
    const time = f.timestamp ? new Date(f.timestamp).toLocaleTimeString() : '';
    const source = f.source_role === 'user' ? '👤 user' : '🤖 assistant';
    const superseded = f.superseded ? ' superseded' : '';
    const supersededBadge = f.superseded ? '<span class="superseded-badge">SUPERSEDED</span>' : '';
    return `<div class="feed-item type-${type}${superseded}"><div class="feed-meta"><span><span class="feed-type ${type}">${type}</span>${supersededBadge}<span class="feed-source">${source}</span></span><span class="feed-time">${time}</span></div><div class="feed-text">${escapeHtml(f.text)}</div>${entities.length ? `<div class="feed-entities">${entities.map(e => `<span class="feed-entity-tag">${e}</span>`).join('')}</div>` : ''}</div>`;
  }).join('');
}

function updateDecisions(data) {
  const container = document.getElementById('decisions-list');
  const decisions = data.decisions || [];
  if (!decisions.length) { container.innerHTML = '<div style="color:var(--text-dim);font-size:11px;padding:8px 0">No decisions recorded</div>'; return; }
  container.innerHTML = decisions.slice(-5).reverse().map(d => `<div class="recall-item"><div class="recall-query">✅ ${escapeHtml(d.text).substring(0,100)}</div><div class="recall-results">${d.date}</div></div>`).join('');
}

function updateTasks(data) {
  const container = document.getElementById('tasks-list');
  const tasks = data.tasks || [];
  if (!tasks.length) { container.innerHTML = '<div style="color:var(--text-dim);font-size:11px;padding:8px 0">No open tasks</div>'; return; }
  container.innerHTML = tasks.slice(-5).reverse().map(t => `<div class="recall-item"><div class="recall-query" style="color:var(--amber)">⏳ ${escapeHtml(t.text).substring(0,100)}</div><div class="recall-results">${t.date}</div></div>`).join('');
}

function updateTypeBreakdown(facts) {
  const counts = {};
  facts.forEach(f => { const t = f.type || 'general'; counts[t] = (counts[t] || 0) + 1; });
  const total = facts.length || 1;
  const colors = {decision:'var(--green)',task:'var(--amber)',preference:'var(--purple)',quantitative:'var(--cyan)',general:'var(--text-dim)'};
  const types = ['quantitative','general','decision','task','preference'];
  document.getElementById('type-breakdown').innerHTML = types.map(t => {
    const count = counts[t] || 0;
    const pct = (count / total * 100).toFixed(0);
    return `<div class="type-bar"><span class="type-bar-label">${t}</span><div class="type-bar-track"><div class="type-bar-fill" style="width:${pct}%;background:${colors[t]||'var(--text-dim)'}"></div></div><span class="type-bar-count">${count}</span></div>`;
  }).join('');
}

function escapeHtml(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }

setInterval(() => {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const mins = Math.floor(elapsed / 60); const secs = elapsed % 60;
  document.getElementById('uptime').textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}, 1000);
setInterval(fetchData, POLL_INTERVAL);
fetchData();
</script>
</body>
</html>"""


# ── Vault scanner (reads markdown files for the dashboard API) ──

def _scan_all_vault_facts(vault_path: str) -> list[dict]:
    """Scan all vault markdown files and return parsed facts."""
    facts = []
    memory_dir = os.path.join(vault_path, "memory")
    if not os.path.exists(memory_dir):
        return facts

    type_map = {"decisions": "decision", "tasks": "task", "preferences": "preference"}

    for root, dirs, files in os.walk(memory_dir):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            rel = os.path.relpath(filepath, memory_dir)
            folder = rel.split(os.sep)[0] if os.sep in rel else ""

            with open(filepath, "r") as fh:
                for line in fh:
                    is_superseded = line.startswith("- ~~[")
                    if not line.startswith("- [") and not is_superseded:
                        continue
                    if "] " not in line:
                        continue

                    raw_line = line
                    if is_superseded:
                        raw_line = line.replace("~~", "").strip()
                        raw_line = re.sub(r'\s*\*\(superseded[^)]*\)\*', '', raw_line)

                    date_part = raw_line.split("] ")[0].replace("- [", "").replace("- ~~[", "").strip()
                    text_part = "] ".join(raw_line.split("] ")[1:]).strip()

                    entities = re.findall(r'\[\[[^\]|]+\|([^\]]+)\]\]', text_part)
                    clean_text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text_part)

                    if folder in type_map:
                        fact_type = type_map[folder]
                    elif "$" in text_part or "%" in text_part:
                        fact_type = "quantitative"
                    else:
                        fact_type = "general"

                    facts.append({
                        "text": clean_text, "type": fact_type, "source_role": "user",
                        "timestamp": f"{date_part}T00:00:00", "entities": entities,
                        "quantities": [], "source_file": rel,
                        "superseded": is_superseded,
                    })

    seen = set()
    unique = []
    for f in facts:
        key = f["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    unique.sort(key=lambda f: f["timestamp"])
    return unique


# ── API handlers ──

def create_app(vault_path: str, db_path: str = None) -> web.Application:
    """Create the dashboard aiohttp application."""
    vault_path = os.path.expanduser(vault_path)

    async def handle_index(request):
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')

    async def handle_stats(request):
        try:
            facts = _scan_all_vault_facts(vault_path)
            sessions_dir = os.path.join(vault_path, "memory", "sessions")
            sessions = sorted(Path(sessions_dir).glob("*.md")) if os.path.exists(sessions_dir) else []
            session_id = sessions[-1].stem if sessions else "none"
            entities_dir = os.path.join(vault_path, "memory", "entities")
            entity_count = len([f for f in os.listdir(entities_dir) if f.endswith(".md")]) if os.path.exists(entities_dir) else 0
            superseded_count = sum(1 for f in facts if f.get("superseded"))
            return web.json_response({
                "session_id": session_id, "total_memories": len(facts),
                "ingested_turns": len(set(f["timestamp"][:16] for f in facts)),
                "stored_facts": len(facts), "deduped_facts": 0,
                "superseded_facts": superseded_count, "entity_count": entity_count,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_entities(request):
        entities_dir = os.path.join(vault_path, "memory", "entities")
        if not os.path.exists(entities_dir):
            return web.json_response({"entities": [], "count": 0})
        entities = []
        for f in sorted(os.listdir(entities_dir)):
            if f.endswith(".md"):
                filepath = os.path.join(entities_dir, f)
                name = f.replace(".md", "").replace("-", " ")
                with open(filepath, "r") as fh:
                    lines = [l for l in fh.readlines() if l.startswith("- [")]
                entities.append({"name": name, "file": f, "fact_count": len(lines)})
        return web.json_response({"entities": entities, "count": len(entities)})

    async def handle_decisions(request):
        filepath = os.path.join(vault_path, "memory", "decisions", "decisions.md")
        if not os.path.exists(filepath):
            return web.json_response({"decisions": [], "count": 0})
        with open(filepath, "r") as fh:
            lines = [l.strip() for l in fh.readlines() if l.startswith("- [")]
        decisions = []
        for line in lines:
            if "] " in line:
                date_part = line.split("] ")[0].replace("- [", "")
                text_part = "] ".join(line.split("] ")[1:])
                decisions.append({"date": date_part, "text": text_part})
        return web.json_response({"decisions": decisions, "count": len(decisions)})

    async def handle_tasks(request):
        filepath = os.path.join(vault_path, "memory", "tasks", "tasks.md")
        if not os.path.exists(filepath):
            return web.json_response({"tasks": [], "count": 0})
        with open(filepath, "r") as fh:
            lines = [l.strip() for l in fh.readlines() if l.startswith("- [")]
        tasks = []
        for line in lines:
            if "] " in line:
                date_part = line.split("] ")[0].replace("- [", "")
                text_part = "] ".join(line.split("] ")[1:])
                tasks.append({"date": date_part, "text": text_part})
        return web.json_response({"tasks": tasks, "count": len(tasks)})

    async def handle_feed(request):
        try:
            return web.json_response(_scan_all_vault_facts(vault_path))
        except Exception:
            return web.json_response([], status=200)

    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/api/stats', handle_stats)
    app.router.add_get('/api/entities', handle_entities)
    app.router.add_get('/api/decisions', handle_decisions)
    app.router.add_get('/api/tasks', handle_tasks)
    app.router.add_get('/api/feed', handle_feed)
    return app


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default="~/phantom_vault")
    parser.add_argument("--port", type=int, default=8422)
    args = parser.parse_args()

    vault = os.path.expanduser(args.vault)
    app = create_app(vault_path=vault)
    print(f"◈ Phantom Dashboard — http://localhost:{args.port}")
    web.run_app(app, host='localhost', port=args.port, print=lambda *a: None)
