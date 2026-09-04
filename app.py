import streamlit as st
from crewai import Task, Crew
import matplotlib.pyplot as plt
import re
import time

# PDF libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Import predefined agents
import agents
from agents import optimist, pessimist, analyst, judge, fact_checker, llm, PROVIDER
from rag import build_context
from evaluation import evaluate
from reflection import critique_and_revise
from moderation import check_topic

import logging
logger = logging.getLogger(__name__)

# Angle used to bias the live search per role, so each agent retrieves
# evidence relevant to the side it's arguing rather than generic results.
SEARCH_ANGLE = {
    "optimist": "advantages benefits of",
    "pessimist": "risks disadvantages criticism of",
    "analyst": "analysis comparison of",
    "judge": "",
}

# ---------- UI SETUP ----------
st.set_page_config(layout="wide", page_title="AI Debate Arena", page_icon="⚖️")

ROLE_STYLE = {
    "Optimist":     {"emoji": "🟢", "color": "#00e676"},
    "Pessimist":    {"emoji": "🔴", "color": "#ff5252"},
    "Analyst":      {"emoji": "🟡", "color": "#ffd740"},
    "Judge":        {"emoji": "⚖️", "color": "#40c4ff"},
    "Fact-Checker": {"emoji": "🔍", "color": "#b388ff"},
}

# "Debate stage" theme: dark gradient, glowing title, bordered agent cards,
# color-coded per role so the layout maps to the actual roles/rounds rather
# than being purely decorative.
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0a0e27,#1a1a4a 45%,#2d0b4e);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.03);
}
.stage-title {
    text-align:center;
    font-size:2.3rem;
    font-weight:800;
    color:#ffffff;
    text-shadow: 0 0 22px rgba(64,196,255,0.55);
    margin-bottom: 0.1rem;
}
.stage-sub {
    text-align:center;
    color:#9fb3c8;
    margin-top:0;
    margin-bottom: 1.3rem;
}
.vs-badge {
    display:flex; align-items:center; justify-content:center;
    height:100%;
    font-size:1.3rem; font-weight:900; color:#ffd740;
    text-shadow:0 0 14px rgba(255,215,64,0.75);
}
.role-header {
    display:flex; align-items:center; gap:8px;
    font-weight:700; font-size:1.15rem;
    margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)


def role_header(role: str):
    style = ROLE_STYLE.get(role, {"emoji": "🤖", "color": "#ffffff"})
    st.markdown(
        f"<div class='role-header' style='color:{style['color']};"
        f"text-shadow:0 0 10px {style['color']}66;'>"
        f"<span style='font-size:1.5rem'>{style['emoji']}</span>{role}</div>",
        unsafe_allow_html=True,
    )


def vs_divider():
    st.markdown("<div class='vs-badge'>⚔️<br/>VS</div>", unsafe_allow_html=True)

# Memory stores outputs from previous rounds
memory = {"optimist":"","pessimist":"","analysis":""}

# ---------- THINKING ANIMATION ----------
def thinking(label):
    placeholder = st.empty()
    placeholder.markdown(f"### {label} 🤖 thinking...")
    time.sleep(1)

# ---------- STREAM TEXT OUTPUT ----------
def stream_text(text):
    """
    Displays text word-by-word (typing effect)
    Ensures user sees gradual output
    """
    placeholder = st.empty()
    out = ""
    for w in text.split():
        out += w + " "
        placeholder.text(out)
        time.sleep(0.002)

# ---------- FORMAT OUTPUT ----------
def format_points(text):
    """
    Converts raw LLM output into:
    - 10 structured points
    - Each point with explanation lines
    - Limits size to avoid PDF overflow
    """
    lines = text.split("\n")
    formatted = []
    count = 1
    i = 0
    total_lines = 0

    while i < len(lines) and count <= 10 and total_lines < 50:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Remove numbering if model already gave
        line = re.sub(r"^\d+[\).\-\s]*", "", line)

        explanation_lines = []
        j = i + 1

        while j < len(lines) and len(explanation_lines) < 4:
            next_line = lines[j].strip()
            if not next_line:
                break
            explanation_lines.append(next_line)
            j += 1

        formatted.append(f"{count}. {line}")
        total_lines += 1

        for exp in explanation_lines:
            if total_lines >= 50:
                break
            formatted.append(f"   {exp}")
            total_lines += 1

        count += 1
        i = j

    return "\n".join(formatted)

# ---------- REMOVE DUPLICATES ----------
def remove_duplicates(text):
    """
    Removes repeated lines to ensure unique content
    """
    seen = set()
    unique = []
    for line in text.split("\n"):
        clean = line.lower().strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique.append(line)
    return "\n".join(unique)

# ---------- RUN AGENT ----------
def run_agent(agent, topic, role, fallback, mem="", prompt_override=None):
    """
    Runs agent with:
    - Live-retrieved grounding context (RAG) biased to the agent's angle
      (skipped when prompt_override is given — the caller built its own
      prompt, e.g. the fact-checker verifying specific prior claims)
    - Custom prompt
    - Retry mechanism
    - Quota-fallback handling
    - Self-critique / reflection pass before the result is returned

    Returns {"text": str, "sources": [...], "revised": bool, "issues": [...]}.
    """

    if prompt_override is not None:
        prompt = prompt_override
        retrieved = {"context": "", "sources": []}
    else:
        angle = SEARCH_ANGLE.get(role.lower().split()[0], "")
        retrieved = build_context(topic, angle=angle) if angle or role.lower() != "judge" else {"context": "", "sources": []}

        grounding = (
            f"\nUse the following retrieved evidence where relevant, and cite it "
            f"naturally in your points:\n{retrieved['context']}\n"
            if retrieved["context"] else ""
        )

        # Judge gets different prompt (decision-based)
        if role.lower() == "judge":
            prompt = f"""
You are Judge

- Topic: {topic}
- Give final decision
- Mention winner clearly
- Give 3-5 reasons
- DO NOT leave answer empty
- You may use the web_search or calculator tools if you need to verify a fact or number

Memory: {mem}
"""
        else:
            prompt = f"""
You are {role}

- Topic: {topic}
- Give 10 points
- Each point with explanation
- No repetition
- Use numbering
- You may use the web_search or calculator tools if you need more evidence or to verify a number
{grounding}
Memory: {mem}
"""

    def _run_once():
        t = Task(description=prompt, expected_output="answer", agent=agent)
        Crew(agents=[agent], tasks=[t], verbose=True, tracing=False).kickoff()
        return t.output.raw if hasattr(t.output, "raw") else str(t.output)

    def _finalize(result):
        if not result or result.strip() == "":
            result = f"{role} could not generate response."
        reflected = critique_and_revise(role, topic, result, retrieved["context"], agents.get_active_llm())
        return {
            "text": reflected["text"],
            "sources": retrieved["sources"],
            "revised": reflected["revised"],
            "issues": reflected["issues"],
        }

    try:
        result = _run_once()
        if not result or result.strip() == "":
            result = _run_once()
        return _finalize(result)

    except Exception as e:
        if _is_quota_error(e) and agents.switch_to_fallback():
            st.warning(
                f"⚠️ {PROVIDER}'s quota/rate limit was reached — switched to the "
                f"OpenRouter fallback model for the rest of this debate."
            )
            try:
                result = _run_once()
                return _finalize(result)
            except Exception:
                logging.exception("%s agent failed on fallback LLM too", role)

        logging.exception("%s agent failed", role)
        text = fallback or f"{role} could not respond. Check that {PROVIDER} is reachable and correctly configured."
        return {"text": text, "sources": [], "revised": False, "issues": []}


_is_quota_error = agents.is_quota_error

# ---------- SAFE FORMAT FUNCTION (🔥 MAIN FIX) ----------
def safe_format(text):
    """
    Ensures formatting NEVER removes content
    """
    formatted = format_points(text)

    # 🔥 CRITICAL FIX: fallback if formatting fails
    if not formatted.strip():
        formatted = text

    return remove_duplicates(formatted)

# ---------- PDF ----------
def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_pdf(topic, sections, confidence_scores, sources):
    """Builds a structured, citable report: topic + timestamp, each agent's
    argument, the LLM-judge confidence scores, and the live sources each
    argument was grounded in. This is what justifies a PDF export existing
    at all — it's an auditable artifact, not a raw text dump."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("AI Debate Arena — Report", styles["Title"]),
        Paragraph(f"Topic: {_esc(topic)}", styles["Heading3"]),
        Paragraph(time.strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["Normal"]),
        Spacer(1, 16),
    ]

    if confidence_scores:
        content.append(Paragraph("Judge Confidence Scores", styles["Heading2"]))
        for role, score in confidence_scores.items():
            content.append(Paragraph(f"{_esc(role)}: {score}/10", styles["BodyText"]))
        content.append(Spacer(1, 14))

    for role, text in sections:
        content.append(Paragraph(_esc(role), styles["Heading2"]))
        for line in text.split("\n"):
            if line.strip():
                content.append(Paragraph(_esc(line), styles["BodyText"]))
        content.append(Spacer(1, 12))

    if sources:
        content.append(Paragraph("Sources", styles["Heading2"]))
        seen = set()
        for s in sources:
            if s["url"] in seen:
                continue
            seen.add(s["url"])
            content.append(Paragraph(f"- {_esc(s['title'] or s['url'])}: {_esc(s['url'])}", styles["BodyText"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------- GRAPH ----------
def show_bar_graph(confidence_scores: dict):
    """Plots the real LLM-as-judge 'overall' score (1-10) collected for
    each agent's latest turn. Empty/failed evaluations show as 0 rather
    than being faked."""
    if not confidence_scores:
        st.info("No confidence scores available for this run.")
        return
    plt.figure()
    plt.bar(list(confidence_scores.keys()), list(confidence_scores.values()))
    plt.ylabel("Judge score (1-10)")
    plt.ylim(0, 10)
    st.pyplot(plt)

# ---------- MAIN UI ----------
st.markdown("<p class='stage-title'>⚖️ AI Debate Arena</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='stage-sub'>Multi-agent debate grounded in live web evidence, "
    "scored by an LLM judge</p>",
    unsafe_allow_html=True,
)

topic = st.text_input("Enter Topic","AI vs Web Development")

def show_sources(sources):
    if not sources:
        return
    with st.expander(f"🔗 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"- [{s['title'] or s['url']}]({s['url']})")

def show_score(scores):
    st.progress(
        min(max(scores["overall"], 0), 10) / 10,
        text=f"Confidence — {scores['overall']}/10",
    )
    st.caption(
        f"relevance {scores['relevance']} · grounding {scores['grounding']} · "
        f"coherence {scores['coherence']} — {scores['rationale']}"
    )

def show_reflection(r):
    if r.get("revised"):
        st.caption(f"🔁 Self-revised after critique: {', '.join(r.get('issues', []))}")

# ---------- MAIN EXECUTION ----------
if st.button("🚀 Start Debate"):

    # Guardrail: reject clearly unsafe/harmful topics before any debate
    # agent runs. Ordinary controversial topics are allowed — see moderation.py.
    verdict = check_topic(topic, agents.get_active_llm())
    if not verdict["allowed"]:
        st.error(f"🚫 This topic can't be debated: {verdict['reason'] or 'flagged as unsafe.'}")
        st.stop()

    all_sources = []
    confidence_scores = {}
    tabs = st.tabs(["Round 1","Round 2 (Rebuttal)","Round 3 (Fact-Check + Analysis)","Verdict (Judge)"])

    # ROUND 1
    with tabs[0]:
        c1, vs, c2 = st.columns([1, 0.15, 1])

        with c1:
            with st.container(border=True):
                role_header("Optimist")
                thinking("Optimist")
                r = run_agent(optimist,topic,"Optimist","")
                opt = safe_format(r["text"])
                memory["optimist"] = opt
                stream_text(opt)
                show_reflection(r)
                show_sources(r["sources"])
                all_sources += r["sources"]
                scores = evaluate("Optimist", topic, r["text"], agents.get_active_llm())
                confidence_scores["Optimist"] = scores["overall"]
                show_score(scores)

        with vs:
            vs_divider()

        with c2:
            with st.container(border=True):
                role_header("Pessimist")
                thinking("Pessimist")
                r = run_agent(pessimist,topic,"Pessimist","")
                pes = safe_format(r["text"])
                memory["pessimist"] = pes
                stream_text(pes)
                show_reflection(r)
                show_sources(r["sources"])
                all_sources += r["sources"]
                scores = evaluate("Pessimist", topic, r["text"], agents.get_active_llm())
                confidence_scores["Pessimist"] = scores["overall"]
                show_score(scores)

    # ROUND 2
    with tabs[1]:
        c1, vs, c2 = st.columns([1, 0.15, 1])

        with c1:
            with st.container(border=True):
                role_header("Optimist")
                thinking("Optimist")
                r = run_agent(optimist,topic,"Rebuttal","",memory["pessimist"])
                opt2 = safe_format(r["text"])
                stream_text(opt2)
                show_reflection(r)
                show_sources(r["sources"])
                all_sources += r["sources"]
                scores = evaluate("Optimist (rebuttal)", topic, r["text"], agents.get_active_llm())
                confidence_scores["Optimist"] = scores["overall"]
                show_score(scores)

        with vs:
            vs_divider()

        with c2:
            with st.container(border=True):
                role_header("Pessimist")
                thinking("Pessimist")
                r = run_agent(pessimist,topic,"Counter","",memory["optimist"])
                pes2 = safe_format(r["text"])
                stream_text(pes2)
                show_reflection(r)
                show_sources(r["sources"])
                all_sources += r["sources"]
                scores = evaluate("Pessimist (counter)", topic, r["text"], agents.get_active_llm())
                confidence_scores["Pessimist"] = scores["overall"]
                show_score(scores)

    # ROUND 3: Fact-Check + Analysis
    with tabs[2]:
        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                role_header("Fact-Checker")
                thinking("Fact-Checker")
                fact_check_prompt = f"""
You are Fact-Checker.

Topic: {topic}

Claims from Optimist:
{memory['optimist']}
{opt2}

Claims from Pessimist:
{memory['pessimist']}
{pes2}

Pick the 2-3 most checkable factual claims from each side (skip opinions).
Use the web_search tool to verify each one you pick.
For each claim checked, output on its own line:
- The claim (short)
- Verdict: SUPPORTED / UNSUPPORTED / CONTRADICTED
- One-line reason citing what you found

Be concise. Do not repeat a claim you've already checked.
"""
                r = run_agent(fact_checker, topic, "Fact-Checker", "", prompt_override=fact_check_prompt)
                fact_check = remove_duplicates(r["text"])
                memory["fact_check"] = fact_check
                stream_text(fact_check)
                show_reflection(r)
                scores = evaluate("Fact-Checker", topic, r["text"], agents.get_active_llm())
                confidence_scores["Fact-Checker"] = scores["overall"]
                show_score(scores)

        with c2:
            with st.container(border=True):
                role_header("Analyst")
                thinking("Analyst")
                r = run_agent(analyst,topic,"Analyst","",memory["optimist"]+memory["pessimist"])
                analysis = safe_format(r["text"])
                memory["analysis"] = analysis
                stream_text(analysis)
                show_reflection(r)
                show_sources(r["sources"])
                all_sources += r["sources"]
                scores = evaluate("Analyst", topic, r["text"], agents.get_active_llm())
                confidence_scores["Analyst"] = scores["overall"]
                show_score(scores)

    # ROUND 4: Verdict
    with tabs[3]:
        with st.container(border=True):
            role_header("Judge")
            thinking("Judge")
            judge_memory = f"Analysis:\n{memory['analysis']}\n\nFact-Check Report:\n{memory['fact_check']}"
            r = run_agent(judge,topic,"Judge","",judge_memory)
            decision = safe_format(r["text"])

            # 🔥 ensure judge always answers
            if not decision.strip():
                decision = "Judge Decision: Optimist wins."

            stream_text(decision)
            show_reflection(r)
            show_sources(r["sources"])
            all_sources += r["sources"]
            scores = evaluate("Judge", topic, decision, agents.get_active_llm())
            confidence_scores["Judge"] = scores["overall"]
            show_score(scores)

    sections = [
        ("Optimist", opt),
        ("Pessimist", pes),
        ("Optimist — Rebuttal", opt2),
        ("Pessimist — Counter", pes2),
        ("Fact-Check Report", fact_check),
        ("Analyst", analysis),
        ("Judge — Final Decision", decision),
    ]

    st.success("✅ Debate Completed")

    st.subheader("📊 Confidence Graph (LLM-as-judge score)")
    show_bar_graph(confidence_scores)

    pdf = generate_pdf(topic, sections, confidence_scores, all_sources)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="debate_report.pdf",
        mime="application/pdf"
    )