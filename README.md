# 🤖 Agentic AI Debate System
Agentic AI Debate System using CrewAI and Streamlit

An interactive **Agentic AI-powered debate platform** built using **CrewAI and Streamlit**, where multiple intelligent agents simulate a structured debate and provide a final decision.
##video on project:https://youtu.be/4otUibIDY_4
---

## 📌 Overview

This project demonstrates the concept of **Agentic AI**, where multiple autonomous agents collaborate, argue, analyze, and make decisions.

The system simulates a debate between:
- 🟢 Optimist (positive arguments)
- 🔴 Pessimist (critical arguments)
- 🟡 Analyst (neutral evaluation)
- 🔵 Judge (final decision maker)

Each agent works independently but shares context through memory, creating a realistic multi-agent discussion system.

---

## 🎯 Objectives

- To implement a **multi-agent AI system**
- To simulate **real-world debate and decision-making**
- To demonstrate **agent collaboration and reasoning**
- To build an **interactive UI using Streamlit**
- To generate structured outputs and reports

---

## ⚙️ Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Core programming |
| Streamlit | UI development |
| CrewAI | Multi-agent orchestration |
| Groq API (LLaMA 3.1) | LLM backend |
| Matplotlib | Visualization |
| ReportLab | PDF generation |

---

## 🧠 System Architecture


User Input (Topic)
↓
Optimist Agent → Gives advantages
↓
Pessimist Agent → Gives disadvantages
↓
Analyst Agent → Compares both
↓
Judge Agent → Final decision
↓
Output (UI + PDF + Graph)


---

## 🔄 Workflow

1. User enters a topic
2. Round 1:
   - Optimist gives positive points
   - Pessimist gives negative points
3. Round 2:
   - Both agents counter each other
4. Round 3:
   - Analyst evaluates arguments
   - Judge gives final verdict
5. Results displayed with:
   - Structured text
   - Confidence graph
   - Downloadable PDF report

---

## 🧩 Features

✅ Multi-agent debate system  
✅ Structured point-wise answers  
✅ Rebuttal mechanism  
✅ Memory-based reasoning  
✅ Final decision (Judge)  
✅ Confidence graph visualization  
✅ PDF report generation  
✅ Smooth UI with animations  
✅ Duplicate removal & formatting  
✅ Safe fallback handling  

---

## 🔐 Agent Roles

### 🟢 Optimist
- Focuses on advantages
- Provides positive perspective
- Supports the topic

### 🔴 Pessimist
- Focuses on disadvantages
- Provides critical analysis
- Challenges the topic

### 🟡 Analyst
- Compares both sides
- Provides balanced evaluation
- Uses logical reasoning

### 🔵 Judge
- Makes final decision
- Declares winner
- Justifies reasoning

---

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-ai-debate.git
cd agentic-ai-debate
2. Install Dependencies
pip install -r requirements.txt
3. Add API Key

Replace in app.py:

api_key="YOUR_API_KEY"
▶️ Run the Project
streamlit run app.py
