# Clinical Decision Making and Pattern Recognition
### Cotiviti Intern Assessment — Hackathon Proof of Concept

**Topic:** Clinical Decision Making and Pattern Recognition in Health Care  
**Author:** Sai Deepa Kadaru  
**Video Demo:** https://www.loom.com/share/f2e57c2f91f7467dba458a58a7ebce4f  
**Stack:** Python, LangGraph, LangChain, OpenAI, Streamlit

---

## Overview

This project demonstrates an agentic AI pipeline for clinical decision making and pattern recognition across the three core domains of healthcare operations: Treatment, Payment, and Operations (TPO).

A healthcare query is submitted to a multi-agent LangGraph system that classifies the query, routes it to a specialized domain expert agent, and produces a structured decision with chain-of-thought reasoning, risk assessment, and recommended actions. A second LLM-as-Judge pass then independently evaluates the quality of that decision on three dimensions.

---

## Architecture

```
User Query
    |
    v
Validation Gate (gpt-4o-mini)
    |-- not healthcare --> reject with reason
    |
    v
Router Agent (gpt-4o-mini)
    |
    |-- treatment  --> Treatment Agent
    |-- payment    --> Payment Agent
    |-- operations --> Operations Agent
    |
    v
Structured Decision Output
    |
    v
LLM-as-Judge Evaluator (gpt-5-nano)
    |
    v
Evaluation Scores (Domain Accuracy / Regulatory Alignment / Completeness)
```

---

## Features

- Multi-agent routing via LangGraph state machine
- Domain-specialized chain-of-thought reasoning for Treatment, Payment, and Operations scenarios
- Structured output using Pydantic models (decision, risk level, recommended actions)
- Input validation gate that rejects non-healthcare queries before they reach the pipeline
- LLM-as-Judge evaluation using a separate model (gpt-5-nano) scoring the decision on three independent dimensions
- Risk level classification (low / medium / high) with domain-specific criteria for each level
- Streamlit UI styled with Cotiviti brand colors

---

## File Structure

```
cotiviti_clinical/
├── app.py              # Streamlit UI and application entry point
├── graph.py            # LangGraph StateGraph definition and workflow
├── agents.py           # Router, domain agent nodes, and validation gate
├── prompts.py          # Prompt templates for all agents
├── evaluator.py        # LLM-as-Judge evaluation module
├── anomaly_chart.py    # Time-series anomaly detection utility (Plotly)
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/saideepa05/cotiviti_clinical.git
cd cotiviti_clinical
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## How It Works

**Step 1 — Validation**  
The input is first checked by a validation agent to confirm it describes a legitimate healthcare scenario. Non-healthcare inputs (greetings, test strings, off-topic queries) are rejected before entering the pipeline.

**Step 2 — Routing**  
A router agent classifies the validated query into one of three domains:
- Treatment: clinical decisions, prior authorization, medical necessity
- Payment: claims review, billing compliance, fraud/waste/abuse detection
- Operations: capacity management, staffing, patient throughput

**Step 3 — Domain Agent Decision**  
The specialized domain agent applies step-by-step chain-of-thought reasoning following evidence-based criteria (CMS guidelines, clinical standards, regulatory benchmarks) and returns a structured decision with risk level and recommended actions.

**Step 4 — LLM-as-Judge Evaluation**  
A separate, independent model (gpt-5-nano) evaluates the decision on three dimensions:
- Domain Accuracy: correctness and relevance of the reasoning to the specific scenario
- Regulatory Alignment: adherence to healthcare regulations, clinical guidelines, and compliance standards
- Completeness: specificity and sufficiency of the recommended actions

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph |
| LLM framework | LangChain |
| Generation model | OpenAI gpt-4o-mini |
| Evaluation model | OpenAI gpt-5-nano |
| Structured output | Pydantic |
| UI | Streamlit |
| Environment | python-dotenv |

---

## Topic Coverage

This POC demonstrates the following concepts from the assessment topic:

| Concept | Demonstrated By |
|---------|----------------|
| Agentic Generative AI | LangGraph multi-agent state machine |
| Chain Reasoning | Step-by-step chain-of-thought in each domain agent |
| Classification | Router agent classifying queries into TPO domains |
| Inference | Domain agents inferring decisions from clinical/billing/operational evidence |
| Pattern Recognition | Risk classification with domain-specific criteria for low/medium/high |

---


