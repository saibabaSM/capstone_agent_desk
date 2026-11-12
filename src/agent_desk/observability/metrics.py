"""Prometheus metrics exposed by the FastAPI app at /metrics (scraped by Prometheus -> Grafana).

Dashboard panels (deploy/grafana/dashboards/analyst_desk.json) and the alert
(deploy/prometheus/alerts.yml: verified share < 98%) are built on these names.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

NOTES = Counter("agentdesk_notes_total", "Notes finished, by outcome", ["outcome"])
NUMBERS = Counter("agentdesk_numbers_total", "Numbers in drafted notes, by verification result", ["verified"])
VERIFIED_SHARE = Gauge("agentdesk_verified_share", "Share of numbers verified in the most recent note")
CRITIC_REJECTIONS = Counter("agentdesk_critic_rejections_total", "Drafts sent back by the compliance critic")
TOKENS = Counter("agentdesk_tokens_total", "LLM tokens", ["role", "kind"])
COST = Counter("agentdesk_cost_usd_total", "LLM cost in USD", ["role"])
NOTE_COST = Histogram("agentdesk_note_cost_usd", "Cost per note in USD",
                      buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))
RUN_LATENCY = Histogram("agentdesk_run_latency_seconds", "Seconds from request to note ready for approval",
                        buckets=(1, 5, 10, 20, 40, 60, 120, 240, 480))
TOOL_CALLS = Counter("agentdesk_tool_calls_total", "MCP tool calls", ["server", "tool", "status"])
TOOL_LATENCY = Histogram("agentdesk_tool_latency_seconds", "MCP tool call latency", ["server"],
                         buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30))
EDGAR_CACHE = Counter("agentdesk_edgar_cache_total", "EDGAR cache lookups", ["result"])
GUARD_BLOCKS = Counter("agentdesk_guard_blocks_total", "Requests, tool calls or passages blocked", ["kind"])
APPROVALS = Counter("agentdesk_approvals_total", "Approval decisions", ["decision"])
REPLANS = Counter("agentdesk_replans_total", "Plans revised after contradicting evidence")
