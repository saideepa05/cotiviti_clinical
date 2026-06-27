from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts import ROUTER_PROMPT, TREATMENT_PROMPT, PAYMENT_PROMPT, OPERATIONS_PROMPT

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ValidationResult(BaseModel):
    is_healthcare: bool
    reason: str


_validation_chain = (
    ChatPromptTemplate.from_template(
        """You are a gatekeeper for a healthcare AI system.

Determine if the following input is a genuine healthcare scenario related to:
- Clinical treatment, diagnosis, or prior authorization
- Medical billing, coding, or claims
- Hospital or health system operations

Input: {query}

If it is a real healthcare query, set is_healthcare=true.
If it is a greeting, test input, gibberish, or unrelated topic, set is_healthcare=false and briefly explain why."""
    )
    | llm.with_structured_output(ValidationResult)
)


def validate_query(query: str) -> ValidationResult:
    return _validation_chain.invoke({"query": query})


class RouterOutput(BaseModel):
    domain: Literal["treatment", "payment", "operations"]
    rationale: str


class AgentOutput(BaseModel):
    chain_of_thought: str
    decision: str
    risk_level: Literal["low", "medium", "high"]
    recommended_actions: list[str]


_router_chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm.with_structured_output(RouterOutput)
_treatment_chain = ChatPromptTemplate.from_template(TREATMENT_PROMPT) | llm.with_structured_output(AgentOutput)
_payment_chain = ChatPromptTemplate.from_template(PAYMENT_PROMPT) | llm.with_structured_output(AgentOutput)
_operations_chain = ChatPromptTemplate.from_template(OPERATIONS_PROMPT) | llm.with_structured_output(AgentOutput)


def router_node(state: dict) -> dict:
    result: RouterOutput = _router_chain.invoke({"query": state["query"]})
    return {"domain": result.domain, "router_rationale": result.rationale}


def treatment_node(state: dict) -> dict:
    result: AgentOutput = _treatment_chain.invoke({"query": state["query"]})
    return {
        "chain_of_thought": result.chain_of_thought,
        "decision": result.decision,
        "risk_level": result.risk_level,
        "recommended_actions": result.recommended_actions,
    }


def payment_node(state: dict) -> dict:
    result: AgentOutput = _payment_chain.invoke({"query": state["query"]})
    return {
        "chain_of_thought": result.chain_of_thought,
        "decision": result.decision,
        "risk_level": result.risk_level,
        "recommended_actions": result.recommended_actions,
    }


def operations_node(state: dict) -> dict:
    result: AgentOutput = _operations_chain.invoke({"query": state["query"]})
    return {
        "chain_of_thought": result.chain_of_thought,
        "decision": result.decision,
        "risk_level": result.risk_level,
        "recommended_actions": result.recommended_actions,
    }
