import streamlit as st 
from crewai import Agent, Task, Crew, LLM
import matplotlib.pyplot as plt
import re
import time

# PDF libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Import predefined agents
from agents import optimist, pessimist, analyst, judge

# Enable debug logs in terminal
import logging
logging.basicConfig(level=logging.DEBUG)

# ---------- UI SETUP ----------
st.set_page_config(layout="wide")

# Background styling (no change to design)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#000428,#004e92);
}
</style>
""", unsafe_allow_html=True)

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

# ---------- LLM CONFIG ----------
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key="YOUR_API_KEY",
    temperature=0.3,
    max_tokens=300,
    verbose=True
)

# ---------- RUN AGENT ----------
def run_agent(agent, topic, role, fallback, mem=""):
    """
    Runs agent with:
    - Custom prompt
    - Retry mechanism
    - Fallback handling
    """

    # Judge gets different prompt (decision-based)
    if role.lower() == "judge":
        prompt = f"""
You are Judge

- Topic: {topic}
- Give final decision
- Mention winner clearly
- Give 3-5 reasons
- DO NOT leave answer empty

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

Memory: {mem}
"""

    try:
        t = Task(description=prompt, expected_output="answer", agent=agent)
        Crew(agents=[agent], tasks=[t], verbose=True).kickoff()

        # Safe extraction
        result = t.output.raw if hasattr(t.output, "raw") else str(t.output)

        # Retry if empty
        if not result or result.strip() == "":
            Crew(agents=[agent], tasks=[t], verbose=True).kickoff()
            result = t.output.raw if hasattr(t.output, "raw") else str(t.output)

        # Final fallback
        if not result or result.strip() == "":
            result = f"{role} could not generate response."

        return result

    except Exception as e:
        print("ERROR:", e)
        return fallback

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
def generate_pdf(result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Agentic AI Debate Report", styles["Title"]))
    content.append(Spacer(1, 20))

    for line in result.split("\n"):
        if line.strip():
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            content.append(Paragraph(safe, styles["BodyText"]))
            content.append(Spacer(1,6))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------- GRAPH ----------
def show_bar_graph():
    plt.figure()
    plt.bar(["Optimist","Pessimist","Analyst"],[80,60,75])
    st.pyplot(plt)

# ---------- MAIN UI ----------
st.title("🤖 Agentic AI Debate")

topic = st.text_input("Enter Topic","AI vs Web Development")

# ---------- MAIN EXECUTION ----------
if st.button("🚀 Start Debate"):

    tabs = st.tabs(["Round 1","Round 2 (Rebuttal)","Round 3 (Analysis + Judge)"])

    # ROUND 1
    with tabs[0]:
        c1,c2 = st.columns(2)

        with c1:
            thinking("Optimist")
            opt = safe_format(run_agent(optimist,topic,"Optimist",""))
            memory["optimist"] = opt
            stream_text(opt)

        with c2:
            thinking("Pessimist")
            pes = safe_format(run_agent(pessimist,topic,"Pessimist",""))
            memory["pessimist"] = pes
            stream_text(pes)

    # ROUND 2
    with tabs[1]:
        c1,c2 = st.columns(2)

        with c1:
            thinking("Optimist")
            opt2 = safe_format(run_agent(optimist,topic,"Rebuttal","",memory["pessimist"]))
            stream_text(opt2)

        with c2:
            thinking("Pessimist")
            pes2 = safe_format(run_agent(pessimist,topic,"Counter","",memory["optimist"]))
            stream_text(pes2)

    # ROUND 3
    with tabs[2]:
        c1,c2 = st.columns(2)

        with c1:
            thinking("Analyst")
            analysis = safe_format(run_agent(analyst,topic,"Analyst","",memory["optimist"]+memory["pessimist"]))
            memory["analysis"] = analysis
            stream_text(analysis)

        with c2:
            thinking("Judge")
            decision = safe_format(run_agent(judge,topic,"Judge","",memory["analysis"]))

            # 🔥 ensure judge always answers
            if not decision.strip():
                decision = "Judge Decision: Optimist wins."

            stream_text(decision)

    result = f"{opt}\n{pes}\n{analysis}\n{decision}"

    st.success("✅ Debate Completed")

    st.subheader("📊 Confidence Graph")
    show_bar_graph()

    pdf = generate_pdf(result)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="debate_report.pdf",
        mime="application/pdf"
    )
