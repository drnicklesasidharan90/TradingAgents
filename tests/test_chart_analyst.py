from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from cli.models import AnalystType
from cli.utils import ANALYST_ORDER
from tradingagents.agents.analysts.chart_analyst import create_chart_analyst
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class DummyChartLLM:
    def bind_tools(self, tools):
        return RunnableLambda(lambda _: AIMessage(content="Chart report", tool_calls=[]))


def test_chart_analyst_writes_chart_report_without_tool_calls():
    chart = create_chart_analyst(DummyChartLLM())
    state = {
        "messages": [HumanMessage(content="NVDA")],
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
    }

    result = chart(state)

    assert result["sender"] == "Chart Analyst"
    assert result["chart_report"] == "Chart report"
    assert result["messages"][0].content == "Chart report"


def test_chart_analyst_supported_in_state_cli_and_conditional_logic():
    state = Propagator().create_initial_state("NVDA", "2026-01-15")
    logic = ConditionalLogic()

    assert state["chart_report"] == ""
    assert AnalystType.CHART.value == "chart"
    assert ("Chart Analyst", AnalystType.CHART) in ANALYST_ORDER
    assert logic.should_continue_chart({"messages": [AIMessage(content="done")]}) == "Msg Clear Chart"
