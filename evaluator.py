from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


class EvaluationScores(BaseModel):
    domain_accuracy: int = Field(
        description="0-100: How accurately does the decision address the specific domain issue?"
    )
    regulatory_alignment: int = Field(
        description="0-100: How well does the decision align with healthcare regulations, CMS guidelines, or clinical evidence?"
    )
    completeness: int = Field(
        description="0-100: How complete and actionable are the decision and recommended actions?"
    )
    evaluation_notes: str = Field(
        description="1-2 sentence critical summary of decision quality and any gaps."
    )


JUDGE_PROMPT = """You are a senior AI quality evaluator at Cotiviti, a healthcare payment accuracy company.

Evaluate the AI-generated healthcare decision below on three dimensions. Score each 0-100.
Be critical and realistic — a score above 85 means the decision is genuinely strong.

Original Query: {query}
Domain: {domain}
AI Decision: {decision}
Recommended Actions:
{actions}

Scoring:
- domain_accuracy: Is the reasoning sound and directly appropriate for this {domain} scenario?
- regulatory_alignment: Does the decision reference or align with relevant regulations, CMS guidelines, clinical evidence, or compliance standards?
- completeness: Are the recommended actions specific, actionable, and sufficient to resolve the issue?"""

_judge_chain = (
    ChatPromptTemplate.from_template(JUDGE_PROMPT)
    | llm.with_structured_output(EvaluationScores)
)


def evaluate_decision(
    query: str,
    domain: str,
    decision: str,
    actions: list[str],
) -> EvaluationScores:
    return _judge_chain.invoke({
        "query":    query,
        "domain":   domain,
        "decision": decision,
        "actions":  "\n".join(f"- {a}" for a in actions),
    })
