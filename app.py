import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from graph import graph
from evaluator import evaluate_decision
from agents import validate_query

# Cotiviti brand colors (from cotiviti.com)
PURPLE   = "#4B0082"   # primary nav purple
ACCENT   = "#D91C5C"   # logo pink-red accent
CYAN     = "#00BCD4"   # tech graphic cyan

SAMPLE_QUERIES = {
    "Treatment — Prior Auth Review (High Risk)": (
        "65-year-old patient with Type 2 diabetes (HbA1c 9.2%), BMI 42, and hypertension "
        "requesting prior authorization for bariatric surgery. Patient has failed two "
        "supervised diet and exercise programs over the past 18 months."
    ),
    "Payment — Claim Integrity (High Risk)": (
        "Claim submitted for cardiac catheterization (CPT 93458) billed at $12,800. "
        "Medicare patient, procedure performed in outpatient setting. Prior authorization "
        "was not obtained. Primary diagnosis code: I25.10 (atherosclerotic heart disease)."
    ),
    "Operations — Capacity Crisis (High Risk)": (
        "ICU bed utilization has been at 94% for 6 consecutive days. Nursing staff overtime "
        "is running at 38%. Elective surgical schedule has not been adjusted. Three patients "
        "are currently awaiting ICU step-down to med-surg beds."
    ),
    "Treatment — Routine Visit (Low Risk)": (
        "28-year-old healthy patient requesting prior authorization for a routine annual "
        "wellness exam and seasonal flu vaccination. No chronic conditions, no current "
        "medications, no recent hospitalizations. Primary care physician referral on file."
    ),
    "Payment — Routine Claim (Low Risk)": (
        "Claim submitted for a routine office visit (CPT 99213) for a 35-year-old patient "
        "with a minor upper respiratory infection. Billed at $145. Primary insurance active, "
        "no prior authorization required, diagnosis code J06.9 matches the procedure."
    ),
    "Operations — Routine Scheduling (Low Risk)": (
        "Outpatient scheduling coordinator requests guidance on optimal appointment slot "
        "allocation for routine follow-up visits next week. Current occupancy is at 68%, "
        "staff levels are fully covered, and no elective procedures are backlogged."
    ),
}

RISK_CONFIG = {
    "low":    {"color": "#2E7D32", "label": "LOW",    "icon": "✅"},
    "medium": {"color": "#E65100", "label": "MEDIUM", "icon": "⚠️"},
    "high":   {"color": "#B71C1C", "label": "HIGH",   "icon": "🔴"},
}

DOMAIN_CONFIG = {
    "treatment":  {"label": "TREATMENT",  "color": PURPLE},
    "payment":    {"label": "PAYMENT",    "color": ACCENT},
    "operations": {"label": "OPERATIONS", "color": CYAN},
}

st.set_page_config(
    page_title="ClinicalAI Decision Support | Cotiviti",
    layout="wide",
)

# Cotiviti brand CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: #ffffff; }}
    header[data-testid="stHeader"] {{ background-color: {PURPLE}; }}
    div.stButton > button[kind="primary"] {{
        background-color: {PURPLE}; border-color: {PURPLE};
        color: #ffffff; font-weight: 600; border-radius: 4px;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #3a0066; border-color: #3a0066; color: #ffffff;
    }}
    h1 {{ color: {PURPLE}; font-weight: 700; }}
    h2, h3 {{ color: {PURPLE}; }}
    hr {{ border-top: 2px solid {PURPLE}22; }}
    div[data-testid="stInfo"] {{
        background-color: #f4f0fa; border-left-color: {PURPLE}; color: #1a1a1a;
    }}
    details summary {{ color: {PURPLE}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# Branded header
st.markdown(f"""
<div style="background:{PURPLE};padding:18px 24px;border-radius:6px;margin-bottom:8px;
            display:flex;align-items:center;gap:16px;">
    <span style="color:white;font-size:22px;font-weight:800;letter-spacing:2px;">
        COT<span style="color:{ACCENT};">I</span>VITI
    </span>
    <span style="color:#ffffff99;font-size:14px;border-left:1px solid #ffffff44;padding-left:16px;">
        Clinical Decision Support · Agentic AI POC
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Clinical Decision Making & Pattern Recognition")
st.markdown(
    "**Agentic AI for Healthcare Treatment, Payment & Operations (TPO)**  \n"
    "*LangGraph multi-agent chain reasoning · GPT-4o-mini*"
)
st.divider()

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Query Input")

    choice = st.selectbox(
        "Load a sample scenario",
        ["— select a sample —"] + list(SAMPLE_QUERIES.keys()),
    )

    prefill = SAMPLE_QUERIES.get(choice, "")
    query = st.text_area(
        "Or enter your own healthcare query",
        value=prefill,
        height=200,
        placeholder="Describe a clinical, billing, or operational scenario...",
    )

    run = st.button("Analyze with AI", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**How it works**")
    st.markdown(f"""
<ol style="padding-left:18px;line-height:1.8">
  <li><b style="color:{PURPLE}">Router Agent</b> classifies the query into Treatment, Payment, or Operations</li>
  <li><b style="color:{PURPLE}">Domain Expert Agent</b> applies step-by-step chain reasoning</li>
  <li>Returns a structured decision, risk level, and action items</li>
</ol>
<p style="color:#888;font-size:12px;margin-top:8px">Built with LangGraph · LangChain · OpenAI</p>
""", unsafe_allow_html=True)

with right:
    st.subheader("AI Decision Output")

    if run and query.strip():
        with st.spinner("Validating query..."):
            validation = validate_query(query)

        if not validation.is_healthcare:
            st.error(
                f"This does not appear to be a healthcare scenario and cannot be processed.  \n"
                f"**Reason:** {validation.reason}  \n\n"
                f"Please describe a clinical, billing, or operational healthcare situation."
            )
            st.stop()

        with st.spinner("Running multi-agent analysis + evaluation..."):
            try:
                result = graph.invoke({"query": query})
                evaluation = evaluate_decision(
                    query=query,
                    domain=result.get("domain", ""),
                    decision=result.get("decision", ""),
                    actions=result.get("recommended_actions", []),
                )
            except Exception as e:
                st.error(f"Error running graph: {e}")
                st.stop()

        domain = result.get("domain", "")
        dcfg   = DOMAIN_CONFIG.get(domain, {"label": domain.upper(), "color": "#333333"})
        risk   = result.get("risk_level", "low")
        rcfg   = RISK_CONFIG.get(risk, RISK_CONFIG["low"])

        d_color = dcfg["color"]
        d_label = dcfg["label"]
        r_color = rcfg["color"]
        r_icon  = rcfg["icon"]
        r_label = rcfg["label"]

        # Domain + Risk badges
        col_d, col_r = st.columns(2)
        with col_d:
            st.markdown(
                f"<div style='padding:12px 16px;border-radius:6px;"
                f"background:{d_color}15;border-left:4px solid {d_color}'>"
                f"<div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Domain</div>"
                f"<div style='font-size:18px;font-weight:700;color:{d_color}'>{d_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_r:
            st.markdown(
                f"<div style='padding:12px 16px;border-radius:6px;"
                f"background:{r_color}15;border-left:4px solid {r_color}'>"
                f"<div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Risk Level</div>"
                f"<div style='font-size:18px;font-weight:700;color:{r_color}'>{r_icon} {r_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        if result.get("router_rationale"):
            st.caption(f"Routing rationale: {result['router_rationale']}")

        st.divider()

        st.markdown("**Decision**")
        st.info(result.get("decision", "No decision returned."))

        with st.expander("Chain of Thought Reasoning", expanded=True):
            st.markdown(result.get("chain_of_thought", ""))

        st.markdown("**Recommended Actions**")
        for i, action in enumerate(result.get("recommended_actions", []), 1):
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;border-radius:4px;"
                f"background:#f9f9f9;border-left:3px solid {d_color}'>"
                f"<b style='color:{d_color}'>{i}.</b> {action}</div>",
                unsafe_allow_html=True,
            )

        # ── LLM-as-Judge Evaluation ───────────────────────────────────────────
        st.divider()
        st.markdown(
            f"**AI Decision Quality Evaluation** "
            f"<span style='font-size:11px;background:{PURPLE};color:white;"
            f"padding:2px 8px;border-radius:10px;margin-left:6px'>LLM-as-Judge</span>",
            unsafe_allow_html=True,
        )

        def _score_bar(label: str, score: int):
            bar_color = "#2E7D32" if score >= 75 else "#E65100" if score >= 50 else "#B71C1C"
            st.markdown(
                f"<div style='margin-bottom:12px'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                f"<span style='font-size:13px;font-weight:600;color:#333'>{label}</span>"
                f"<span style='font-size:13px;font-weight:700;color:{bar_color}'>{score}/100</span>"
                f"</div>"
                f"<div style='background:#e0e0e0;height:10px;border-radius:5px'>"
                f"<div style='background:{bar_color};width:{score}%;height:10px;border-radius:5px'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            _score_bar("Domain Accuracy",       evaluation.domain_accuracy)
        with c2:
            _score_bar("Regulatory Alignment",  evaluation.regulatory_alignment)
        with c3:
            _score_bar("Completeness",          evaluation.completeness)

        st.caption(f"Evaluator notes: {evaluation.evaluation_notes}")

    elif run:
        st.warning("Please enter a query or select a sample scenario.")
    else:
        st.markdown(f"""
<div style="background:#f9f6ff;border-radius:8px;padding:20px;border:1px solid {PURPLE}22">
  <p style="color:#666;margin-bottom:16px">
    Select a sample query from the left panel or type your own scenario,
    then click <b>Analyze with AI</b> to see the agentic decision pipeline in action.
  </p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:{PURPLE};color:white">
      <th style="padding:8px 12px;text-align:left">Domain</th>
      <th style="padding:8px 12px;text-align:left">Example Scenario</th>
    </tr>
    <tr style="border-bottom:1px solid #eee">
      <td style="padding:8px 12px;color:{PURPLE};font-weight:600">Treatment</td>
      <td style="padding:8px 12px;color:#444">Prior authorization, medical necessity, clinical guidelines</td>
    </tr>
    <tr style="border-bottom:1px solid #eee;background:#fafafa">
      <td style="padding:8px 12px;color:{ACCENT};font-weight:600">Payment</td>
      <td style="padding:8px 12px;color:#444">Claims review, coding accuracy, fraud/waste/abuse detection</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;color:{CYAN};font-weight:600">Operations</td>
      <td style="padding:8px 12px;color:#444">Capacity management, staffing, patient throughput</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)
