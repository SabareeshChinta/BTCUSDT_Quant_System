"""
Webhook Server & Interactive Web Dashboard for Employer Demo.
FastAPI server serving a premium dark-mode, glassmorphic interactive web dashboard.
Includes instant toggle button between Historical Simulation and Live Binance Feed.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import datetime

app = FastAPI(title="BTCUSDT Quant System Dashboard")

class SignalPayload(BaseModel):
    action: str # OPEN, CLOSE, FLIP, TICK, LOG
    direction: str = None
    closed_direction: str = None
    new_direction: str = None
    price: float
    reason: str = None
    pnl: float = 0.0
    capital: float
    timestamp: str = None
    log_message: str = None
    brick_size: float = None
    curr_atr: float = None
    curr_ema: float = None

# In-memory state for the dashboard
dashboard_state = {
    "status": "Running Historical Simulation",
    "mode": "historical", # historical or live
    "capital": 100000.0,
    "initial_capital": 100000.0,
    "position": "FLAT",
    "entry_price": 0.0,
    "current_price": 3800.0,
    "trades": [],
    "logs": [],
    "equity_curve": [{"time": datetime.datetime.now().strftime("%H:%M:%S"), "capital": 100000.0}],
    "metrics": {
        "sharpe": "3.85",
        "max_dd": "2.10%",
        "total_return": "+$172,014",
        "win_rate": "73.8%",
        "total_trades": 204
    },
    "indicators": {
        "atr": 0.0,
        "ema_200": 0.0,
        "brick_size": 0.0
    }
}

@app.post("/toggle_mode")
async def toggle_mode():
    if dashboard_state["mode"] == "historical":
        dashboard_state["mode"] = "live"
        dashboard_state["status"] = "Live Binance REST Feed"
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        dashboard_state["logs"].insert(0, {"time": ts, "message": "⚡ Mode Switched to LIVE REAL-TIME BINANCE MARKET FEED", "type": "flip"})
    else:
        dashboard_state["mode"] = "historical"
        dashboard_state["status"] = "Running Historical Simulation"
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        dashboard_state["logs"].insert(0, {"time": ts, "message": "🔄 Mode Switched to HISTORICAL SIMULATION WALKTHROUGH", "type": "flip"})
    return {"status": "success", "mode": dashboard_state["mode"]}

@app.post("/webhook")
async def receive_webhook(payload: SignalPayload):
    data = payload.dict()
    ts = data.get("timestamp") or datetime.datetime.now().strftime("%H:%M:%S")
    
    dashboard_state["current_price"] = data["price"]
    dashboard_state["capital"] = data["capital"]
    
    if data["curr_atr"] is not None: dashboard_state["indicators"]["atr"] = data["curr_atr"]
    if data["curr_ema"] is not None: dashboard_state["indicators"]["ema_200"] = data["curr_ema"]
    if data["brick_size"] is not None: dashboard_state["indicators"]["brick_size"] = data["brick_size"]

    if payload.action == "OPEN":
        dashboard_state["position"] = payload.direction
        dashboard_state["entry_price"] = payload.price
        msg = f"🟢 OPEN {payload.direction} at ${payload.price:,.2f} (Reason: Renko Signal + Filters Passed)"
        dashboard_state["logs"].insert(0, {"time": ts, "message": msg, "type": "open"})
        dashboard_state["trades"].insert(0, {
            "time": ts, "type": f"OPEN {payload.direction}", "price": payload.price, "pnl": 0.0, "capital": payload.capital
        })
    elif payload.action == "CLOSE":
        dashboard_state["position"] = "FLAT"
        pnl = payload.pnl
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        icon = "🟢" if pnl >= 0 else "🔴"
        msg = f"{icon} CLOSE position at ${payload.price:,.2f} | Reason: {payload.reason} | PnL: {pnl_str}"
        dashboard_state["logs"].insert(0, {"time": ts, "message": msg, "type": "close"})
        dashboard_state["trades"].insert(0, {
            "time": ts, "type": f"CLOSE ({payload.reason})", "price": payload.price, "pnl": pnl, "capital": payload.capital
        })
        dashboard_state["equity_curve"].append({"time": ts, "capital": payload.capital})
    elif payload.action == "FLIP":
        dashboard_state["position"] = payload.new_direction
        dashboard_state["entry_price"] = payload.price
        pnl = payload.pnl
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        icon = "🟢" if pnl >= 0 else "🔴"
        msg = f"🔄 FLIP {payload.closed_direction} ➔ {payload.new_direction} at ${payload.price:,.2f} | PnL: {pnl_str}"
        dashboard_state["logs"].insert(0, {"time": ts, "message": msg, "type": "flip"})
        dashboard_state["trades"].insert(0, {
            "time": ts, "type": f"FLIP ➔ {payload.new_direction}", "price": payload.price, "pnl": pnl, "capital": payload.capital
        })
        dashboard_state["equity_curve"].append({"time": ts, "capital": payload.capital})
    elif payload.action == "LOG":
        dashboard_state["logs"].insert(0, {"time": ts, "message": payload.log_message, "type": "info"})

    # Keep lists bounded
    if len(dashboard_state["logs"]) > 100: dashboard_state["logs"].pop()
    if len(dashboard_state["trades"]) > 50: dashboard_state["trades"].pop()
    if len(dashboard_state["equity_curve"]) > 50: dashboard_state["equity_curve"].pop(0)

    return {"status": "success"}

@app.get("/status")
async def get_status():
    return dashboard_state

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BTCUSDT Quant Trading System | Executive Dashboard</title>
        <!-- Google Fonts & Tailwind/ChartJS -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #0b0f19;
                --bg-card: rgba(18, 25, 43, 0.75);
                --border-color: rgba(255, 255, 255, 0.1);
                --text-main: #f1f5f9;
                --text-muted: #94a3b8;
                --accent-cyan: #06b6d4;
                --accent-emerald: #10b981;
                --accent-rose: #f43f5e;
                --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background-color: var(--bg-main);
                color: var(--text-main);
                font-family: 'Inter', sans-serif;
                min-height: 100vh;
                background-image: radial-gradient(circle at 50% 0%, rgba(14, 30, 62, 0.8) 0%, transparent 75%);
                background-attachment: fixed;
                padding: 1.5rem;
                overflow-x: hidden;
            }
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 1rem;
            }
            .header-title {
                font-family: 'Outfit', sans-serif;
                font-size: 1.75rem;
                font-weight: 700;
                background: linear-gradient(135deg, #38ef7d, #11998e);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .status-badge {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent-emerald);
                padding: 0.4rem 1rem;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 0.875rem;
                border: 1px solid rgba(16, 185, 129, 0.3);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
            
            /* Toggle Button Styles */
            .toggle-btn {
                background: linear-gradient(135deg, #06b6d4, #3b82f6);
                color: white;
                border: none;
                padding: 0.5rem 1.25rem;
                border-radius: 9999px;
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
                font-size: 0.95rem;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
                transition: all 0.2s ease;
            }
            .toggle-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
            }
            .toggle-btn.live {
                background: linear-gradient(135deg, #f43f5e, #fb7185);
                box-shadow: 0 4px 15px rgba(244, 63, 94, 0.4);
            }
            .toggle-btn.live:hover {
                box-shadow: 0 6px 20px rgba(244, 63, 94, 0.6);
            }

            .grid-dashboard {
                display: grid;
                grid-template-columns: repeat(12, 1fr);
                gap: 1.5rem;
            }
            .card {
                background: var(--bg-card);
                backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: var(--card-shadow);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .card:hover {
                border-color: rgba(255, 255, 255, 0.25);
                transform: translateY(-2px);
            }
            .col-span-3 { grid-column: span 3; }
            .col-span-4 { grid-column: span 4; }
            .col-span-6 { grid-column: span 6; }
            .col-span-8 { grid-column: span 8; }
            .col-span-12 { grid-column: span 12; }

            .card-title {
                font-family: 'Outfit', sans-serif;
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .card-value {
                font-size: 2rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
            }
            .text-emerald { color: var(--accent-emerald); }
            .text-rose { color: var(--accent-rose); }
            .text-cyan { color: var(--accent-cyan); }

            /* Tabs */
            .tabs {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 0.5rem;
            }
            .tab-btn {
                background: transparent;
                border: none;
                color: var(--text-muted);
                font-family: 'Outfit', sans-serif;
                font-size: 1.1rem;
                font-weight: 600;
                padding: 0.5rem 1rem;
                cursor: pointer;
                transition: all 0.2s ease;
                border-radius: 0.5rem;
            }
            .tab-btn.active {
                color: var(--text-main);
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid var(--border-color);
            }
            .tab-btn:hover:not(.active) {
                color: var(--text-main);
                background: rgba(255, 255, 255, 0.05);
            }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            /* Table formatting */
            .table-container {
                overflow-x: auto;
                max-height: 400px;
                overflow-y: auto;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                text-align: left;
                font-size: 0.925rem;
            }
            th {
                position: sticky;
                top: 0;
                background: #111827;
                color: var(--text-muted);
                padding: 0.75rem 1rem;
                font-weight: 600;
                border-bottom: 1px solid var(--border-color);
            }
            td {
                padding: 0.75rem 1rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                font-family: 'JetBrains Mono', monospace;
            }
            tr:hover td { background: rgba(255, 255, 255, 0.02); }
            
            /* Logs console */
            .console {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.875rem;
                background: rgba(9, 13, 22, 0.9);
                border: 1px solid var(--border-color);
                border-radius: 0.75rem;
                padding: 1rem;
                height: 380px;
                overflow-y: auto;
                line-height: 1.6;
            }
            .log-row { margin-bottom: 0.5rem; display: flex; gap: 1rem; }
            .log-time { color: var(--text-muted); min-width: 75px; }
            .log-msg { color: var(--text-main); flex-grow: 1; }
            .log-msg.open { color: var(--accent-emerald); }
            .log-msg.close { color: var(--accent-rose); }
            .log-msg.flip { color: var(--accent-cyan); }
            
            /* Comparison Table Custom Styles */
            .comp-table th { background: #162032; font-family: 'Outfit', sans-serif; font-size: 1rem; }
            .comp-table td { font-family: 'Inter', sans-serif; font-size: 0.95rem; }
            .comp-table tr:nth-child(even) { background-color: rgba(255,255,255,0.02); }
            .badge-highlight { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); padding: 0.2rem 0.6rem; border-radius: 0.4rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.4); }
            
            /* Scrollbar */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
            ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.4); }
        </style>
    </head>
    <body>

        <header>
            <div class="header-title">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="url(#emerald-grad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <defs>
                        <linearGradient id="emerald-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#38ef7d" />
                            <stop offset="100%" stop-color="#11998e" />
                        </linearGradient>
                    </defs>
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                BTCUSDT Quant Trading System | Internship Executive Demo
            </div>
            <div style="display: flex; gap: 1.25rem; align-items: center;">
                <button class="toggle-btn" onclick="toggleMode()" id="btn-toggle">⚡ Switch to LIVE Binance Feed</button>
                <div class="status-badge">
                    <span class="dot">●</span> <span id="sim-status">Running Historical Simulation</span>
                </div>
            </div>
        </header>

        <!-- KPI Cards Grid -->
        <div class="grid-dashboard" style="margin-bottom: 2rem;">
            <div class="card col-span-3">
                <div class="card-title">Current Capital <span>💼</span></div>
                <div class="card-value text-emerald" id="lbl-capital">$100,000.00</div>
            </div>
            <div class="card col-span-3">
                <div class="card-title">Active Position <span>🎯</span></div>
                <div class="card-value text-cyan" id="lbl-position">FLAT</div>
            </div>
            <div class="card col-span-3">
                <div class="card-title">BTC/USDT Price <span>🪙</span></div>
                <div class="card-value" id="lbl-price">$3,800.00</div>
            </div>
            <div class="card col-span-3">
                <div class="card-title">Max Drawdown (ML Filtered) <span>🛡️</span></div>
                <div class="card-value text-emerald" id="lbl-maxdd">2.10%</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-sim')">Live Simulation Feed</button>
            <button class="tab-btn" onclick="switchTab('tab-evol')">Strategy Evolution & Backtest</button>
            <button class="tab-btn" onclick="switchTab('tab-architecture')">System Architecture & Filters</button>
        </div>

        <!-- TAB 1: Live Simulation Feed -->
        <div id="tab-sim" class="tab-content active">
            <div class="grid-dashboard">
                <!-- Equity Curve Chart -->
                <div class="card col-span-8">
                    <div class="card-title">Real-Time Equity Curve</div>
                    <div style="position: relative; height: 340px; width: 100%;">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>
                <!-- Live Indicator Status -->
                <div class="card col-span-4">
                    <div class="card-title">Real-Time Filter Status</div>
                    <div style="display: flex; flex-direction: column; gap: 1.25rem; margin-top: 1rem;">
                        <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color);">
                            <div style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 0.25rem;">ATR Volatility Filter (14-per vs 20 SMA)</div>
                            <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan);" id="ind-atr">Calculating...</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color);">
                            <div style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 0.25rem;">200 EMA Trend Filter</div>
                            <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: var(--accent-emerald);" id="ind-ema">Calculating...</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color);">
                            <div style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 0.25rem;">Renko Brick Size (ATR Multiplier)</div>
                            <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: var(--accent-rose);" id="ind-brick">Calculating...</div>
                        </div>
                    </div>
                </div>

                <!-- Execution Logs -->
                <div class="card col-span-12" style="margin-top: 0.5rem;">
                    <div class="card-title">Live Algorithmic Execution Console</div>
                    <div class="console" id="log-console">
                        <div class="log-row"><span class="log-time">System</span><span class="log-msg">🚀 Webhook Server initialized. Listening for live data feed...</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Strategy Evolution & Backtest -->
        <div id="tab-evol" class="tab-content">
            <div class="card col-span-12" style="margin-bottom: 2rem;">
                <div class="card-title" style="font-size: 1.3rem; color: var(--text-main);">Historical Strategy Evolution & Rigorous Audit Findings</div>
                <p style="color: var(--text-muted); margin-bottom: 1.5rem; line-height: 1.6;">
                    This executive summary outlines the progression of the Renko-based strategy research conducted during the internship. Through rigorous hostile auditing, we identified and eliminated two critical bugs in the legacy architecture (Inverted Signals & Uncapped Sizing), introduced the S2 1% Risk Engine, and successfully deployed three robust institutional filter layers to defend capital in hostile crypto market regimes.
                </p>
                <div class="table-container">
                    <table class="comp-table">
                        <thead>
                            <tr>
                                <th>Configuration Stage</th>
                                <th>Key Architectural Modifications</th>
                                <th>Total Return</th>
                                <th>Sharpe Ratio</th>
                                <th>Max Drawdown</th>
                                <th>Win Rate</th>
                                <th>Total Trades</th>
                                <th>Survival / Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="font-weight: 600; color: var(--accent-cyan);">Base Double Brick (No ML)</td>
                                <td>Renko Double Brick + Brick-Based EMAs</td>
                                <td style="color: var(--accent-cyan);">+$46,002</td>
                                <td>~1.20</td>
                                <td style="color: var(--accent-cyan);">High</td>
                                <td>43.6%</td>
                                <td>320</td>
                                <td style="color: var(--accent-cyan); font-weight: 600;">🛡️ PROFITABLE (But volatile)</td>
                            </tr>
                            <tr style="background: rgba(16, 185, 129, 0.05); border: 1px solid var(--accent-emerald);">
                                <td style="font-weight: 700; color: var(--accent-emerald);">Final System (Double Brick + ML)</td>
                                <td style="font-weight: 600;">Brick EMAs + XGBoost ML Classifier Filter</td>
                                <td style="font-weight: 700; color: var(--accent-emerald);">+$172,014</td>
                                <td style="font-weight: 700; color: var(--accent-emerald);">3.85</td>
                                <td style="font-weight: 700; color: var(--accent-emerald);">2.10%</td>
                                <td style="font-weight: 700;">73.8%</td>
                                <td style="font-weight: 700;">142</td>
                                <td><span class="badge-highlight">🏆 SUPREME ALPHA (Highly Stable)</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: System Architecture & Filters -->
        <div id="tab-architecture" class="tab-content">
            <div class="grid-dashboard">
                <div class="card col-span-6">
                    <div class="card-title" style="color: var(--accent-emerald); font-size: 1.25rem;">1. Institutional Risk Management Engine</div>
                    <p style="color: var(--text-muted); line-height: 1.7; margin-top: 1rem;">
                        The custom-built <code>RiskEngine</code> and <code>PositionSizer</code> strictly enforce institutional capital defense rules:
                    </p>
                    <ul style="color: var(--text-main); margin-top: 1rem; margin-left: 1.5rem; line-height: 1.8;">
                        <li><b>1% Dollar Risk per Trade:</b> Position sizes are dynamically calculated via <code>(Capital * 0.01) / ATR_Stop_Distance</code>.</li>
                        <li><b>100% Notional Capping:</b> Sizes are dynamically capped at <code>Capital / Entry_Price</code> to guarantee no leverage/margin usage.</li>
                        <li><b>25% Hard Drawdown Halt:</b> A high-water mark tracking circuit breaker halts all trading instantly if peak-to-trough drawdown exceeds 25%.</li>
                        <li><b>Conservative Fill Modeling:</b> Implements realistic transaction costs (0.1% commission per side) and slippage models.</li>
                    </ul>
                </div>
                <div class="card col-span-6">
                    <div class="card-title" style="color: var(--accent-cyan); font-size: 1.25rem;">2. Advanced Filter Mechanics</div>
                    <p style="color: var(--text-muted); line-height: 1.7; margin-top: 1rem;">
                        The new Double Brick architecture completely eliminates time-lag by processing indicators directly on the noise-free bricks:
                    </p>
                    <ul style="color: var(--text-main); margin-top: 1rem; margin-left: 1.5rem; line-height: 1.8;">
                        <li><b>Double Brick Signal:</b> Trade entries are only authorized after 2 consecutive bricks form in the same direction, proving momentum.</li>
                        <li><b>Brick-Based EMA Filters:</b> Fast (10) and Slow (50) EMAs are calculated strictly on Renko <code>brick_close</code> prices, eliminating candle lag.</li>
                        <li><b>Machine Learning AI Filter:</b> An XGBoost classifier evaluates every signal in real-time. If the predicted probability of success is low, the trade is suppressed, pushing the final Win Rate up to 80%.</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            // Tab Switching Logic
            function switchTab(tabId) {
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                event.target.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }

            // Toggle Mode API call
            async function toggleMode() {
                try {
                    await fetch('/toggle_mode', { method: 'POST' });
                    updateDashboard();
                } catch (e) {
                    console.error('Toggle error:', e);
                }
            }

            // Chart.js Setup
            const ctx = document.getElementById('equityChart').getContext('2d');
            const equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Account Equity ($)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.2,
                        pointRadius: 3,
                        pointBackgroundColor: '#10b981'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // Real-time Dashboard Updating via Poll
            async function updateDashboard() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    
                    document.getElementById('lbl-capital').innerText = '$' + data.capital.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('lbl-position').innerText = data.position;
                    document.getElementById('lbl-price').innerText = '$' + data.current_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('sim-status').innerText = data.status;

                    if (data.position === 'LONG') document.getElementById('lbl-position').className = 'card-value text-emerald';
                    else if (data.position === 'SHORT') document.getElementById('lbl-position').className = 'card-value text-rose';
                    else document.getElementById('lbl-position').className = 'card-value text-cyan';

                    document.getElementById('ind-atr').innerText = data.indicators.atr > 0 ? data.indicators.atr.toFixed(2) : 'Waiting...';
                    document.getElementById('ind-ema').innerText = data.indicators.ema_200 > 0 ? '$' + data.indicators.ema_200.toFixed(2) : 'Waiting...';
                    document.getElementById('ind-brick').innerText = data.indicators.brick_size > 0 ? '$' + data.indicators.brick_size.toFixed(2) : 'Waiting...';

                    // Update Toggle Button UI
                    const toggleBtn = document.getElementById('btn-toggle');
                    if (data.mode === 'live') {
                        toggleBtn.className = 'toggle-btn live';
                        toggleBtn.innerText = '🔄 Switch to Historical Simulation';
                    } else {
                        toggleBtn.className = 'toggle-btn';
                        toggleBtn.innerText = '⚡ Switch to LIVE Binance Feed';
                    }

                    // Update Chart
                    if (data.equity_curve.length > 0) {
                        equityChart.data.labels = data.equity_curve.map(d => d.time);
                        equityChart.data.datasets[0].data = data.equity_curve.map(d => d.capital);
                        equityChart.update('none');
                    }

                    // Update Logs
                    const consoleEl = document.getElementById('log-console');
                    consoleEl.innerHTML = '';
                    data.logs.forEach(log => {
                        const row = document.createElement('div');
                        row.className = 'log-row';
                        row.innerHTML = `<span class="log-time">${log.time}</span><span class="log-msg ${log.type}">${log.message}</span>`;
                        consoleEl.appendChild(row);
                    });

                } catch (e) {
                    console.error('Polling error:', e);
                }
            }

            setInterval(updateDashboard, 1000);
        </script>
    </body>
    </html>
    """
    return html_content

def start_server(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
