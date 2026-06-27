from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from agents import router_node, treatment_node, payment_node, operations_node


class ClinicalState(TypedDict):
    query: str
    domain: Literal["treatment", "payment", "operations"]
    router_rationale: str
    chain_of_thought: str
    decision: str
    risk_level: Literal["low", "medium", "high"]
    recommended_actions: list[str]


def _route(state: ClinicalState) -> str:
    return state["domain"]


builder = StateGraph(ClinicalState)

builder.add_node("router", router_node)
builder.add_node("treatment", treatment_node)
builder.add_node("payment", payment_node)
builder.add_node("operations", operations_node)

builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    _route,
    {
        "treatment": "treatment",
        "payment": "payment",
        "operations": "operations",
    },
)
builder.add_edge("treatment", END)
builder.add_edge("payment", END)
builder.add_edge("operations", END)

graph = builder.compile()
