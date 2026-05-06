# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TradingAgents is a multi-agent LLM trading framework built on **LangGraph**. A `TradingAgentsGraph` orchestrates specialized agents (analysts → researchers → trader → risk debators → portfolio manager) that collaboratively produce a trading decision for a given ticker and date. Entry points are the `tradingagents` CLI (`cli.main:app`) and the `TradingAgentsGraph` Python class.

Python `>= 3.10` (project tested against 3.12/3.13). Package layout: `tradingagents/` (library) + `cli/` (Typer/Rich CLI). Both are installed by `pip install .` per `pyproject.toml`.

## Commands

```bash
pip install .                         # install package + console script `tradingagents`
pip install -e .                      # editable install for development

tradingagents                         # interactive CLI (alias of `python -m cli.main`)
tradingagents analyze --checkpoint    # enable LangGraph resume from last successful node
tradingagents analyze --clear-checkpoints  # wipe all per-ticker checkpoint DBs first

python main.py                        # programmatic example (NVDA, fixed date)
python test.py                        # ad-hoc dataflow benchmark (yfinance MACD)

pytest                                # run test suite (testpaths=tests, --strict-markers)
pytest tests/test_signal_processing.py::test_buy   # single test
pytest -m unit                        # markers: unit | integration | smoke
pytest -k checkpoint                  # by name pattern

python scripts/smoke_structured_output.py openai   # exercise structured-output agents
                                                   # against any provider (low cost, no propagate)

docker compose run --rm tradingagents              # containerized run (uses .env)
docker compose --profile ollama run --rm tradingagents-ollama   # local Ollama backend
```

`conftest.py` autouse-injects placeholder values for every supported `*_API_KEY` env var, and provides a `mock_llm_client` fixture that patches `tradingagents.llm_clients.factory.create_llm_client`. Tests therefore run cleanly without real credentials — do not add network or LLM calls to unit tests; mark them `integration` if they need external services.

## Architecture

### Graph pipeline (`tradingagents/graph/`)

`TradingAgentsGraph` (`trading_graph.py`) wires up the LangGraph `StateGraph`:

1. **Analysts** (selectable subset of `["market", "social", "news", "fundamentals"]`) — each runs in a tool-use loop: `<Type> Analyst` ↔ `tools_<type>` until `should_continue_<type>` returns clear, then a `Msg Clear <Type>` node strips intermediate messages before falling through to the next analyst.
2. **Researcher debate** — `Bull Researcher` ↔ `Bear Researcher` for `max_debate_rounds`, then `Research Manager` (deep-think LLM) emits an investment plan via structured output.
3. **Trader** (quick-think LLM) emits a trader decision via structured output.
4. **Risk debate** — `Aggressive` → `Conservative` → `Neutral` rotates for `max_risk_discuss_rounds`, then `Portfolio Manager` (deep-think LLM) issues the final decision via structured output.

The graph is built once in `GraphSetup.setup_graph()` and recompiled with a `SqliteSaver` only when `checkpoint_enabled` is set. `Propagator` builds the initial `AgentState`; `Reflector` runs after deferred outcomes resolve; `SignalProcessor` extracts the canonical Buy/Hold/Sell rating.

### State (`tradingagents/agents/utils/agent_states.py`)

`AgentState` extends `MessagesState` and carries the entire pipeline's artifacts: per-analyst reports (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`), `investment_debate_state` (TypedDict), `investment_plan`, `trader_investment_plan`, `risk_debate_state`, `final_trade_decision`, and `past_context` (memory log injection for the PM prompt). Every node mutates this dict — when adding a new agent you must extend this state and the `_log_state` JSON dump in `trading_graph.py`.

### LLM clients (`tradingagents/llm_clients/`)

`create_llm_client(provider, model, base_url=None, **kwargs)` is the only entry point. Providers in `_OPENAI_COMPATIBLE = (openai, xai, deepseek, qwen, glm, ollama, openrouter)` share the OpenAI-compatible client; `anthropic`, `google`, and `azure` have dedicated clients. **Imports are lazy** in `factory.py` so test collection and CLI startup never pull in unused SDKs. `backend_url` defaults to `None` per provider — never default it to OpenAI's URL (regression risk: it leaks into Gemini/Anthropic and produces malformed URLs).

Provider-specific thinking knobs live in config (`google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`) and are routed by `TradingAgentsGraph._get_provider_kwargs()` into the matching client.

### Structured output (`tradingagents/agents/schemas.py`)

Research Manager, Trader, and Portfolio Manager call `llm.with_structured_output(Schema)` and return typed Pydantic instances rendered back to markdown so memory log / CLI / saved reports keep their existing shape. Provider-native modes are used: `json_schema` (OpenAI/xAI/DeepSeek/Qwen/GLM), `response_schema` (Gemini), tool-use (Anthropic), function-calling (other OpenAI-compatible). 5-tier scale (`Buy/Overweight/Hold/Underweight/Sell`) for Research Manager + Portfolio Manager; 3-tier (`Buy/Hold/Sell`) for Trader. When changing schemas, run `scripts/smoke_structured_output.py <provider>` against each provider you care about.

### Data vendors (`tradingagents/dataflows/`)

`interface.py` exposes a category-aware tool router: every agent-facing tool (`get_stock_data`, `get_indicators`, `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_news`, `get_global_news`, `get_insider_transactions`) dispatches to `VENDOR_METHODS[<tool>][<vendor>]`. Vendor selection comes from `config["data_vendors"]` (per-category default) with `config["tool_vendors"]` overrides per tool. Current vendors: `yfinance` (default, no key) and `alpha_vantage` (requires `ALPHA_VANTAGE_API_KEY`). When adding a vendor, update `VENDOR_LIST`, register implementations in `VENDOR_METHODS`, and ensure `AlphaVantageRateLimitError`-style exceptions remain caught at the boundary.

### Persistence

Two independent persistence layers:

- **Decision log** (always on, `tradingagents/agents/utils/memory.py`): append-only markdown at `~/.tradingagents/memory/trading_memory.md` (override with `TRADINGAGENTS_MEMORY_LOG_PATH`). `propagate()` writes a *pending* entry on completion. The next same-ticker run resolves pending entries via `_fetch_returns()` (raw + alpha vs SPY over 5 holding days) and `Reflector.reflect_on_final_decision()`, then `batch_update_with_outcomes()` commits all reflections in one write. The `_SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"` HTML comment is the entry delimiter — LLMs cannot emit it, so it's a safe parser anchor. `memory_log_max_entries` caps *resolved* entries only; pending entries are never pruned.
- **LangGraph checkpoints** (opt-in, `tradingagents/graph/checkpointer.py`): per-ticker SQLite under `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override base with `TRADINGAGENTS_CACHE_DIR`). `thread_id = sha256(f"{TICKER}:{date}")[:16]` ensures same ticker+date resumes, different date starts fresh. Checkpoints clear on successful completion.

Ticker values flow into filesystem paths in both layers — always pass them through `tradingagents.dataflows.utils.safe_ticker_component()` before joining (rejects path-escape values like `../`).

### Configuration

`tradingagents/default_config.py` is the single source of truth. CLI flow lets users override interactively; programmatic flow is `DEFAULT_CONFIG.copy()` + mutate + pass to `TradingAgentsGraph(config=...)`. Important keys:

- `llm_provider` ∈ `{openai, google, anthropic, xai, deepseek, qwen, glm, openrouter, ollama, azure}`
- `deep_think_llm` (Research Manager, Portfolio Manager) vs `quick_think_llm` (analysts, researchers, trader, risk, signal processor, reflector)
- `max_debate_rounds`, `max_risk_discuss_rounds`, `max_recur_limit`
- `output_language` (analyst reports & final decision; agent debate stays in English)
- `checkpoint_enabled`, `memory_log_path`, `memory_log_max_entries`
- `data_vendors` (category defaults) and `tool_vendors` (per-tool overrides)

`set_config()` from `dataflows.config` must be called whenever config changes — `TradingAgentsGraph.__init__` already does this; if you bypass it, propagate the call yourself.

### CLI (`cli/`)

Typer app with one command (`analyze`) that wraps an interactive Questionary flow. `MessageBuffer` in `cli/main.py` tracks live agent status / report sections — `FIXED_AGENTS`, `ANALYST_MAPPING`, and `REPORT_SECTIONS` mappings drive the Rich layout. `StatsCallbackHandler` (`cli/stats_handler.py`) is passed as a LangChain callback so token / tool-call stats render live.

## Conventions

- **Encoding:** every file I/O passes `encoding="utf-8"` explicitly. The Windows-default `cp1252` corrupts saved reports — preserve this when adding new persistence code.
- **Output language:** `output_language` config affects analyst-report and final-decision *output* only. Internal debate prompts stay English (reasoning quality regressions otherwise).
- **Versioning:** Semantic Versioning within 0.x; breaking changes called out in `CHANGELOG.md`. Update the changelog when shipping user-visible behavior changes.
- **Test markers:** declare `unit`, `integration`, or `smoke` in `pyproject.toml`'s `[tool.pytest.ini_options].markers` — `--strict-markers` is on, so unmarked-but-declared markers fail fast.
- **Dependencies:** `requirements.txt` is intentionally near-empty; `pyproject.toml` is canonical. `uv.lock` is committed.
