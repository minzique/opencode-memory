from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>lume / memory</title>
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #171717;
            --border: #262626;
            --text: #e0e0e0;
            --text-muted: #a3a3a3;
            --accent: #4ade80;
            --accent-dim: rgba(74, 222, 128, 0.1);
            --font-mono: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
        }

        * { box-sizing: border_box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: var(--font-mono);
            font-size: 13px;
            line-height: 1.5;
            padding: 2rem;
        }

        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }

        header {
            display: flex;
            justify_content: space-between;
            align_items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        h1 { font-size: 1.2rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
        h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #fff; }

        .subtitle { color: var(--text-muted); font-size: 0.8rem; }

        .grid {
            display: grid;
            grid-template_columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 1.5rem;
            border-radius: 4px;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .stat-item { display: flex; flex-direction: column; }
        .stat-label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #fff; }

        /* Search */
        .search-container { margin-bottom: 2rem; }
        input[type="text"] {
            width: 100%;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 1rem;
            font-family: var(--font-mono);
            font-size: 1rem;
            border-radius: 4px;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: var(--accent);
        }

        /* Table */
        .table-container {
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--surface);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th, td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            cursor: pointer;
            user-select: none;
        }

        th:hover { color: var(--text); }

        tr:last-child td { border-bottom: none; }
        tr:hover { background: rgba(255, 255, 255, 0.02); }

        .type-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 2px;
            font-size: 0.7rem;
            text-transform: uppercase;
            background: var(--border);
            color: var(--text-muted);
        }

        .type-architecture { background: rgba(59, 130, 246, 0.1); color: #60a5fa; }
        .type-constraint { background: rgba(239, 68, 68, 0.1); color: #f87171; }
        .type-decision { background: rgba(16, 185, 129, 0.1); color: #34d399; }
        .type-fact { background: rgba(107, 114, 128, 0.1); color: #9ca3af; }
        .type-preference { background: rgba(245, 158, 11, 0.1); color: #fbbf24; }
        
        .content-cell {
            max-width: 400px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .expanded .content-cell {
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* States & Episodes */
        .list-item {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .list-item:last-child { border-bottom: none; }
        
        .item-header {
            display: flex;
            justify_content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .item-title { font-weight: 600; color: #fff; }
        .item-meta { color: var(--text-muted); font-size: 0.75rem; }
        
        .progress-bar {
            height: 4px;
            background: var(--border);
            border-radius: 2px;
            margin-top: 0.5rem;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.3s;
        }

        .filters {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .filter-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            cursor: pointer;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            transition: all 0.2s;
        }

        .filter-btn:hover, .filter-btn.active {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-dim);
        }

        .pagination {
            display: flex;
            justify_content: center;
            gap: 1rem;
            margin-top: 1rem;
        }

        .btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-family: var(--font-mono);
            border-radius: 4px;
        }
        .btn:hover { border-color: var(--accent); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .loading { opacity: 0.5; pointer-events: none; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>lume / memory</h1>
            <div class="subtitle" id="version-info">v0.3.0 • <span id="uptime">0s uptime</span></div>
        </div>
        <div style="text-align: right">
            <div class="subtitle" id="connection-status">Connected</div>
        </div>
    </header>

    <div class="grid">
        <div class="panel">
            <h2>System Stats</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <span class="stat-label">Memories</span>
                    <span class="stat-value" id="stat-memories">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Episodes</span>
                    <span class="stat-value" id="stat-episodes">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">States</span>
                    <span class="stat-value" id="stat-states">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">DB Size</span>
                    <span class="stat-value" id="stat-size">0 MB</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Working States</h2>
            <div id="states-list">Loading...</div>
        </div>
    </div>

    <div class="search-container">
        <input type="text" id="search-input" placeholder="Search memories (semantic search)...">
    </div>

    <div class="panel">
        <div style="display: flex; justify_content: space-between; align_items: center; margin-bottom: 1rem;">
            <h2>Memories</h2>
            <div class="filters" id="type-filters">
                <button class="filter-btn active" data-type="all">All</button>
                <button class="filter-btn" data-type="architecture">Arch</button>
                <button class="filter-btn" data-type="constraint">Constraint</button>
                <button class="filter-btn" data-type="decision">Decision</button>
                <button class="filter-btn" data-type="fact">Fact</button>
                <button class="filter-btn" data-type="preference">Pref</button>
                <button class="filter-btn" data-type="convention">Convention</button>
                <button class="filter-btn" data-type="pattern">Pattern</button>
                <button class="filter-btn" data-type="failure">Failure</button>
            </div>
        </div>
        
        <div class="table-container">
            <table id="memories-table">
                <thead>
                    <tr>
                        <th onclick="sortBy('type')">Type</th>
                        <th onclick="sortBy('content')">Content</th>
                        <th onclick="sortBy('scope')">Scope</th>
                        <th onclick="sortBy('confidence')">Conf</th>
                        <th onclick="sortBy('created_at')">Created</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody id="memories-body">
                    <!-- Rows injected via JS -->
                </tbody>
            </table>
        </div>
        <div class="pagination">
            <button class="btn" id="prev-page" onclick="changePage(-1)">Previous</button>
            <span id="page-info" style="align_self: center;">Page 1</span>
            <button class="btn" id="next-page" onclick="changePage(1)">Next</button>
        </div>
    </div>

    <div class="panel" style="margin-top: 2rem;">
        <h2>Recent Episodes</h2>
        <div id="episodes-list">Loading...</div>
    </div>

    <script>
        // State
        let memories = [];
        let currentPage = 1;
        const itemsPerPage = 20;
        let currentFilter = 'all';
        let sortField = 'created_at';
        let sortAsc = false;
        let searchDebounce;

        // Init
        document.addEventListener('DOMContentLoaded', () => {
            fetchStatus();
            fetchStates();
            fetchEpisodes();
            fetchMemories();
            
            setInterval(fetchStatus, 30000);

            // Search listener
            document.getElementById('search-input').addEventListener('input', (e) => {
                clearTimeout(searchDebounce);
                searchDebounce = setTimeout(() => {
                    if (e.target.value.trim()) {
                        searchMemories(e.target.value);
                    } else {
                        fetchMemories();
                    }
                }, 300);
            });

            // Filter listeners
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentFilter = btn.dataset.type;
                    currentPage = 1;
                    renderTable();
                });
            });
        });

        // API Calls
        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                
                document.getElementById('stat-memories').textContent = data.memory_count;
                document.getElementById('stat-episodes').textContent = data.episode_count;
                document.getElementById('stat-states').textContent = data.state_count;
                document.getElementById('stat-size').textContent = (data.db_size_bytes / 1024 / 1024).toFixed(2) + ' MB';
                
                const uptime = Math.floor(data.uptime_seconds);
                const h = Math.floor(uptime / 3600);
                const m = Math.floor((uptime % 3600) / 60);
                document.getElementById('uptime').textContent = `${h}h ${m}m uptime`;
            } catch (e) {
                console.error('Status fetch failed', e);
            }
        }

        async function fetchStates() {
            try {
                const res = await fetch('/states');
                const data = await res.json();
                const container = document.getElementById('states-list');
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="list-item" style="color: var(--text-muted)">No active states</div>';
                    return;
                }

                container.innerHTML = data.map(state => `
                    <div class="list-item">
                        <div class="item-header">
                            <span class="item-title">${state.project_id}</span>
                            <span class="item-meta">${new Date(state.updated_at * 1000).toLocaleDateString()}</span>
                        </div>
                        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">${state.data.objective || 'No objective'}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted)">${state.data.progress || 'No progress recorded'}</div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('States fetch failed', e);
            }
        }

        async function fetchEpisodes() {
            try {
                const res = await fetch('/episodes?limit=5');
                const data = await res.json();
                const container = document.getElementById('episodes-list');
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="list-item" style="color: var(--text-muted)">No episodes recorded</div>';
                    return;
                }

                container.innerHTML = data.map(ep => `
                    <div class="list-item">
                        <div class="item-header">
                            <span class="item-title">${ep.summary}</span>
                            <span class="item-meta">${new Date(ep.created_at * 1000).toLocaleString()}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-muted)">
                            Session: ${ep.session_id.substring(0, 8)}... • 
                            Decisions: ${ep.decisions.length} • 
                            Constraints: ${ep.constraints.length}
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Episodes fetch failed', e);
            }
        }

        async function fetchMemories() {
            try {
                const res = await fetch('/memories?limit=100');
                memories = await res.json();
                renderTable();
            } catch (e) {
                console.error('Memories fetch failed', e);
            }
        }

        async function searchMemories(query) {
            try {
                const res = await fetch('/recall', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, limit: 50 })
                });
                const data = await res.json();
                memories = data.results.map(r => ({ ...r.memory, similarity: r.similarity }));
                currentPage = 1;
                renderTable();
            } catch (e) {
                console.error('Search failed', e);
            }
        }

        // UI Logic
        function renderTable() {
            const tbody = document.getElementById('memories-body');
            let filtered = memories;

            if (currentFilter !== 'all') {
                filtered = memories.filter(m => m.type === currentFilter);
            }

            // Sort
            filtered.sort((a, b) => {
                let valA = a[sortField];
                let valB = b[sortField];
                if (typeof valA === 'string') valA = valA.toLowerCase();
                if (typeof valB === 'string') valB = valB.toLowerCase();
                
                if (valA < valB) return sortAsc ? -1 : 1;
                if (valA > valB) return sortAsc ? 1 : -1;
                return 0;
            });

            // Paginate
            const start = (currentPage - 1) * itemsPerPage;
            const end = start + itemsPerPage;
            const pageItems = filtered.slice(start, end);

            tbody.innerHTML = pageItems.map(m => `
                <tr onclick="this.classList.toggle('expanded')">
                    <td><span class="type-badge type-${m.type}">${m.type}</span></td>
                    <td class="content-cell" title="${m.content}">${m.content}</td>
                    <td>${m.scope}</td>
                    <td>${(m.confidence * 100).toFixed(0)}%</td>
                    <td style="white-space: nowrap; color: var(--text-muted)">
                        ${new Date(m.created_at * 1000).toLocaleDateString()}
                    </td>
                    <td style="color: var(--accent)">${m.similarity ? (m.similarity * 100).toFixed(0) + '%' : ''}</td>
                </tr>
            `).join('');

            document.getElementById('page-info').textContent = `Page ${currentPage} of ${Math.ceil(filtered.length / itemsPerPage) || 1}`;
            document.getElementById('prev-page').disabled = currentPage === 1;
            document.getElementById('next-page').disabled = currentPage >= Math.ceil(filtered.length / itemsPerPage);
        }

        function changePage(delta) {
            currentPage += delta;
            renderTable();
        }

        function sortBy(field) {
            if (sortField === field) {
                sortAsc = !sortAsc;
            } else {
                sortField = field;
                sortAsc = true;
            }
            renderTable();
        }
    </script>
</body>
</html>
    """
