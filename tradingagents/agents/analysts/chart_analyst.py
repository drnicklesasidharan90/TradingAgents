from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)


def create_chart_analyst(llm):
    """Create an agent that reads recent OHLCV data like a trading chart.

    The chart analyst deliberately complements the broader market analyst by
    focusing on price action: trend structure, candlesticks, support/resistance,
    volatility regimes, and volume confirmation.
    """

    def chart_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are an AI chart analyst specializing in trading-chart analysis. Your job is to read recent OHLCV price action as if you were annotating a candlestick chart for a professional trading desk.

Workflow:
1. Call get_stock_data first for the target instrument, using a lookback window that is long enough to inspect swing structure (normally 90 to 180 calendar days ending at the trade date).
2. Use get_indicators selectively to support the chart read. Prefer a compact set such as close_10_ema, close_50_sma, close_200_sma, macd, rsi, boll_ub, boll_lb, atr, and vwma when they are relevant.
3. Do not invent chart levels or patterns. If the data is incomplete, say what is missing and make the most defensible observation from available data.

Your report must cover:
- Trend and market structure: higher highs/lows, lower highs/lows, range behavior, trend breaks, and moving-average alignment.
- Support and resistance: recent swing highs/lows, breakout or breakdown zones, retest areas, and invalidation levels.
- Candlestick and price-action signals: gaps, long wicks, large range candles, consolidation, compression, breakouts, failed breakouts, or reversal clues.
- Momentum and volatility: RSI/MACD behavior, Bollinger band position, ATR expansion/compression, and whether momentum confirms price.
- Volume confirmation: volume spikes, volume-weighted trend confirmation, accumulation/distribution clues, and suspicious low-volume moves.
- Trading plan implications: likely bullish, bearish, and neutral scenarios with entry triggers, stop/reference levels, and risk notes.

Write a detailed, evidence-based chart report for traders. Avoid generic advice; tie every claim to observed price/indicator behavior."""
            + """ Append a Markdown table at the end with columns: Setup, Evidence, Trigger, Invalidation, Risk Note."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "chart_report": report,
            "sender": "Chart Analyst",
        }

    return chart_analyst_node
